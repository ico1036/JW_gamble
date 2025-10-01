#!/usr/bin/env python3
"""
matplotlib 한글 폰트 설정 확인 및 수정
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

print("=" * 80)
print("matplotlib 한글 폰트 설정")
print("=" * 80)

# 시스템 정보
print(f"\n시스템: {platform.system()}")
print(f"matplotlib 버전: {plt.matplotlib.__version__}")

# 사용 가능한 한글 폰트 찾기
print("\n사용 가능한 한글 폰트:")
korean_fonts = []

for font in fm.fontManager.ttflist:
    font_name = font.name
    # macOS 한글 폰트 찾기
    if any(keyword in font_name for keyword in ['Gothic', 'Nanum', 'Malgun', 'Batang', 'Dotum', 'Apple']):
        if font_name not in korean_fonts:
            korean_fonts.append(font_name)

korean_fonts.sort()
for i, font in enumerate(korean_fonts[:20], 1):
    print(f"  {i:2d}. {font}")

# 권장 폰트 찾기
recommended = None
if platform.system() == 'Darwin':  # macOS
    priorities = ['AppleSDGothicNeo-Regular', 'AppleSDGothicNeo', 'AppleGothic', 'Arial Unicode MS']
    for font in priorities:
        if font in korean_fonts:
            recommended = font
            break

print(f"\n권장 폰트: {recommended}")

# 폰트 캐시 위치
cache_dir = fm.get_cachedir()
print(f"\n폰트 캐시 위치: {cache_dir}")

print("\n" + "=" * 80)
print("해결 방법:")
print("=" * 80)
print("1. generate_eda_report.py 파일 수정")
print(f"   plt.rcParams['font.family'] = '{recommended}'")
print("\n2. 폰트 캐시 재생성 (필요시):")
print("   rm -rf ~/.matplotlib")
print("   python -c 'import matplotlib.font_manager; matplotlib.font_manager._rebuild()'")
print("=" * 80)
