__all__ = ['set_korean_font']
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def set_korean_font(font_path: str = None) -> None:
    """
    matplotlib에서 한글 폰트를 설정하고, PDF 저장 시 글꼴이 깨지지 않도록 설정합니다.
    """
    font_name = None  # 최종적으로 사용할 폰트 이름을 저장할 변수

    # 1. OS별로 폰트 경로 탐색 및 폰트 이름 확인
    if os.name == 'nt':  # Windows
        font_path = font_path or 'c:/Windows/Fonts/malgun.ttf'
        try:
            font_prop = fm.FontProperties(fname=font_path)
            font_name = font_prop.get_name()
        except FileNotFoundError:
            print(f"지정된 폰트 파일({font_path})을 찾을 수 없습니다.")
        except Exception as e:
            print(f"폰트 설정 중 오류 발생: {e}")

    elif sys.platform == 'darwin':  # macOS
        font_path = font_path or '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
        try:
            font_prop = fm.FontProperties(fname=font_path)
            font_name = font_prop.get_name()
        except FileNotFoundError:
            print(f"지정된 폰트 파일({font_path})을 찾을 수 없습니다.")
        except Exception as e:
            print(f"폰트 설정 중 오류 발생: {e}")

    elif os.name == 'posix':  # Linux
        try:
            if font_path:
                font_prop = fm.FontProperties(fname=font_path)
                font_name = font_prop.get_name()
            else:
                # 시스템 폰트에서 나눔고딕을 탐색
                font_list = fm.findSystemFonts(fontpaths=['/usr/share/fonts', '/usr/local/share/fonts'], fontext='ttf')
                nanum_fonts = [f for f in font_list if 'NanumGothic' in os.path.basename(f)]
                if not nanum_fonts:
                    raise FileNotFoundError("NanumGothic 계열 폰트를 찾지 못했습니다.")
                font_prop = fm.FontProperties(fname=nanum_fonts[0])
                font_name = font_prop.get_name()
        except FileNotFoundError as e:
            print(f"⚠️ 폰트 설정 오류: {e}")
        except Exception as e:
            print(f"⚠️ 폰트 설정 중 오류 발생: {e}")

    # 2. 폰트 이름이 성공적으로 찾아졌으면, matplotlib의 rcParams를 설정
    if font_name:
        print(f"✅ 적용된 폰트: '{font_name}'")
        plt.rc('font', family=font_name)
        plt.rcParams['axes.unicode_minus'] = False
        
        # PDF/PS 파일 저장 시 폰트 설정
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        print("✅ PDF/PS 폰트 타입 42로 설정 완료 (한글 깨짐 방지)")
    else:
        print("⚠️ 한글 폰트를 찾지 못해 기본 폰트(sans-serif)로 설정합니다.")
        plt.rc('font', family='sans-serif')
        plt.rcParams['axes.unicode_minus'] = False