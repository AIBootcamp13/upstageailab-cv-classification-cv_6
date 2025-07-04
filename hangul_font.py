import matplotlib.pyplot as plt
import platform
from matplotlib import font_manager, rc
import os # os 모듈 임포트

plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

# 폰트 경로 초기화
f_path = None

# 운영체제별 폰트 경로 설정
if platform.system() == 'Darwin': # macOS
    f_path = "/root/upstageailab-cv-classification-cv_6/fonts/NanumGothic.ttf"
elif platform.system() == 'Windows': # Windows
    f_path = "c:/Windows/Fonts/malgun.ttf"
elif platform.system() == 'Linux': # 리눅스 서버 환경
    # >>>>>> 여기에 NanumGothic.ttf 파일의 실제 서버 경로를 입력해주세요 <<<<<<
    # 예시: 사용자 홈 디렉토리 아래의 프로젝트 폴더 내 fonts 디렉토리
    f_path = "/root/upstageailab-cv-classification-cv_6/fonts/NanumGothic.ttf"
    # 또는: 프로젝트 파일과 같은 디렉토리에 업로드했다면 상대 경로 사용 가능
    # f_path = "./fonts/NanumGothic.ttf"
    # 또는: 시스템 폰트 디렉토리에 설치했다면
    # f_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
else:
    print("오류: 지원되지 않는 운영체제입니다. 폰트 경로를 수동으로 설정해야 합니다.")

# 폰트 경로가 설정되었는지, 그리고 파일이 존재하는지 확인
if f_path is None:
    print("오류: 폰트 경로(f_path)가 정의되지 않았습니다. 현재 운영체제에 맞는 경로 설정이 필요합니다.")
elif not os.path.exists(f_path):
    print(f"오류: 지정된 폰트 파일이 '{f_path}' 경로에 존재하지 않습니다. 파일을 확인해주세요.")
    print("리눅스 서버에 폰트 파일을 정확히 업로드했는지, 그리고 경로가 맞는지 다시 확인 부탁드립니다.")
    # 파일이 존재하지 않으면 더 이상 진행할 수 없으므로 여기서 종료하거나 기본 폰트로 폴백합니다.
else:
    print(f"폰트 파일 '{f_path}' 경로 확인 완료.")
    try:
        # Matplotlib 폰트 관리자에 폰트 파일을 명시적으로 추가
        font_manager.fontManager.addfont(f_path)
        # 추가된 폰트의 실제 이름 가져오기
        font_prop = font_manager.FontProperties(fname=f_path)
        font_name = font_prop.get_name()

        # Matplotlib의 기본 폰트 설정
        rc('font', family=font_name)
        print(f"'{font_name}' 폰트가 성공적으로 설정되었습니다!")
    except Exception as e:
        print(f"폰트 설정 중 오류 발생: {e}")
        print("폰트 파일이 손상되었거나, Matplotlib가 해당 폰트를 인식하지 못하는 문제일 수 있습니다.")
