list) / len(accuracy_list) if accuracy_list else 0
    
    return avg_accuracy

# 이미지 시각화 함수
def visualize_prediction(image, boxes, labels, scores, threshold=0.5):
    # 이미지가 텐서라면 넘파이 배열로 변환
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).cpu().numpy()
        # 정규화 복원
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image_np * std + mean
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
    else:
        image_np = np.array(image)
    
    image_pil = Image.fromarray(image_np)
    draw = ImageDraw.Draw(image_pil)
    
    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue
            
        # 박스 그리기
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        
        # 클래스 및 점수 텍스트
        class_name = VOC_CLASS_NAMES[int(label)] if 0 <= int(label) < len(VOC_CLASS_NAMES) else "unknown"
        draw.text((x1, y1), f"{class_name}: {score:.2f}", fill="red")
    
    return image_pil

# 추론 함수
def inference(model, image_path, device, threshold=0.3):
    # 이미지 로드
    image = Image.open(image_path).convert("RGB")
    
    # 전처리 - 리사이징 및 정규화
    input_size = 512
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 추론
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        
        # 후처리
        detections = decode(outputs['hm'], outputs['wh'], outputs['reg'], K=100)[0]
        
        # 결과 처리
        boxes = []
        labels = []
        scores = []
        
        for det in detections:
            if det[4] > threshold:
                x1, y1, x2, y2, score, class_id = det
                boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
                labels.append(int(class_id.item()))
                scores.append(score.item())
    
    # 시각화
    result_image = visualize_prediction(image_tensor[0], torch.tensor(boxes), 
                                       torch.tensor(labels), torch.tensor(scores), threshold)
    
    return result_image, boxes, labels, scores

# 메인 함수
def main():
    # 데이터셋 경로 설정
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    
    # 데이터셋 생성
    try:
        # VOC 2007 데이터셋 다운로드 및 로드
        # trainval 데이터셋 (학습 및 검증)
        train_dataset = VOCDetectionDataset(
            root=data_dir,
            year='2007',
            image_set='trainval',
            download=True
        )
        
        # test 데이터셋
        test_dataset = VOCDetectionDataset(
            root=data_dir,
            year='2007',
            image_set='test',
            download=True
        )
        
        print(f"학습 데이터셋 크기: {len(train_dataset)}")
        print(f"테스트 데이터셋 크기: {len(test_dataset)}")
        
        # 데이터 로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,  # 메모리 사용량 감소를 위해 배치 사이즈 감소
            shuffle=True,
            num_workers=2,
            collate_fn=collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn
        )
        
        # 모델 생성 (더 가벼운 백본 사용)
        num_classes = 20  # VOC 클래스 수 (0부터 19까지)
        model = CenterNet(num_classes=num_classes, backbone_name='resnet18')
        model.to(device)
        
        # 옵티마이저 및 학습률 스케줄러 설정
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=1e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        # 학습 시작
        num_epochs = 30  # VOC 데이터셋에 대해 더 많은 에포크 수행
        best_accuracy = 0.0
        
        # 체크포인트 저장 디렉토리
        os.makedirs('checkpoints', exist_ok=True)
        
        for epoch in range(num_epochs):
            try:
                # 학습
                train_metrics = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
                
                # 학습률 업데이트
                lr_scheduler.step()
                
                # 평가 (각 에포크마다)
                accuracy = evaluate(model, test_loader, device)
                print(f"에포크 {epoch}: Accuracy = {accuracy:.4f}")
                
                # 모델 저장
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                        'best_accuracy': best_accuracy,
                    }, 'checkpoints/best_centernet_voc_model.pth')
                    print(f"새로운 최고 정확도 {best_accuracy:.4f}! 모델이 저장되었습니다.")
                
                # 중간 체크포인트 저장
                if epoch % 5 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    }, f'checkpoints/centernet_voc_checkpoint_epoch_{epoch}.pth')
                    print(f"체크포인트가 저장되었습니다: 에포크 {epoch}")
            except Exception as e:
                print(f"에포크 {epoch} 학습 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()  # 상세 오류 내용 출력
                
                # 중간 체크포인트 저장
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                }, f'checkpoints/centernet_voc_checkpoint_error_epoch_{epoch}.pth')
                print(f"오류 발생으로 인한 체크포인트가 저장되었습니다: 에포크 {epoch}")
                
                # 에러가 발생해도 계속 진행
                continue
    
    except Exception as e:
        print(f"학습 준비 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()  # 상세 오류 내용 출력
        raise e

# 테스트 함수
def test_model():
    # 학습된 모델 로드
    print("학습된 모델을 로드합니다...")
    
    try:
        # 체크포인트 파일 찾기
        checkpoint_files = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
        
        if not checkpoint_files:
            # checkpoints 디렉토리가 없거나 파일이 없는 경우 현재 디렉토리 확인
            checkpoint_files = [f for f in os.listdir('.') if f.endswith('.pth') and f.startswith('centernet_voc')]
        
        if not checkpoint_files:
            print("학습된 모델 파일을 찾을 수 없습니다. 먼저 모델을 학습시켜 주세요.")
            return
        
        # best 모델 찾기, 없으면 가장 최근 체크포인트 사용
        best_model_path = None
        for f in checkpoint_files:
            if 'best' in f:
                best_model_path = os.path.join('checkpoints', f)
                break
        
        if best_model_path is None:
            # 숫자 순으로 정렬하여 가장 최근 체크포인트 선택
            epoch_models = [f for f in checkpoint_files if 'epoch' in f]
            if epoch_models:
                epoch_models.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
                best_model_path = os.path.join('checkpoints', epoch_models[-1])
            else:
                best_model_path = os.path.join('checkpoints', checkpoint_files[0])
        
        print(f"모델 로드 경로: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        
        num_classes = 20  # VOC 클래스 수 (0부터 19까지)
        model = CenterNet(num_classes=num_classes, backbone_name='resnet18')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"최고 정확도: {checkpoint.get('best_accuracy', 0):.4f}, 에포크: {checkpoint.get('epoch', 0)}")
        
        # VOC 데이터셋에서 테스트 이미지 선택
        try:
            test_dataset = VOCDetection(root='./data', year='2007', image_set='test', download=False)
            test_idx = random.randint(0, len(test_dataset) - 1)
            img, annotation = test_dataset[test_idx]
            
            # 임시 파일로 저장
            temp_img_path = 'temp_test_image.jpg'
            img.save(temp_img_path)
            test_image_path = temp_img_path
            
        except Exception as e:
            print(f"VOC 데이터셋 로드 실패: {e}")
            # 임의의 이미지 파일 찾기
            import glob
            all_images = glob.glob('./**/*.jpg', recursive=True)
            if not all_images:
                all_images = glob.glob('./**/*.png', recursive=True)
            
            if not all_images:
                print("테스트할 이미지를 찾을 수 없습니다.")
                return
            
            test_image_path = all_images[0]
        
        print(f"테스트 이미지: {test_image_path}")
        
        # 추론 및 시각화
        result_image, boxes, labels, scores = inference(model, test_image_path, device)
        
        # 결과 저장
        os.makedirs('results', exist_ok=True)
        result_path = 'results/centernet_voc_detection_result.jpg'
        result_image.save(result_path)
        print(f"검출 결과가 '{result_path}'에 저장되었습니다.")
        
        # 검출 결과 출력
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            class_name = VOC_CLASS_NAMES[int(label)] if 0 <= int(label) < len(VOC_CLASS_NAMES) else "unknown"
            print(f"검출 {i+1}: {class_name}, 점수: {score:.4f}, 위치: {box}")
            
        # 임시 파일 삭제
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
    
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e}")
    except Exception as e:
        print(f"모델 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()  # 상세 오류 내용 출력

# 메인 실행
if __name__ == "__main__":
    main()
    test_model()
