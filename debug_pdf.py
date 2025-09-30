#!/usr/bin/env python3
"""
PDF 텍스트 구조 확인용 디버그 스크립트
"""
import pdfplumber

pdf_path = '/Users/ryan/horse_park/2025-09-28.pdf'

with pdfplumber.open(pdf_path) as pdf:
    print(f"총 페이지 수: {len(pdf.pages)}\n")

    for i, page in enumerate(pdf.pages[:2]):  # 첫 2페이지만
        print(f"=== 페이지 {i+1} ===\n")
        text = page.extract_text()

        # 첫 1000자만 출력
        print(text[:2000])
        print("\n" + "="*50 + "\n")

        # 테이블 추출 시도
        tables = page.extract_tables()
        print(f"테이블 개수: {len(tables)}")

        if tables:
            print("\n첫 번째 테이블:")
            for row in tables[0][:5]:  # 첫 5행만
                print(row)