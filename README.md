# GSLIB 세미베리오그램 분석 자동화 스크립트

## 1. 개요

본 파이썬 스크립트 모음은 GSLIB `gamv` 프로그램을 활용한 세미베리오그램 분석 전 과정을 자동화합니다. `.dat` 파일 분석, `gamv.exe` 실행, 결과 시각화 및 이론 모델 피팅을 지원하여 연구 효율성을 높입니다. 모든 스크립트, `gamv.exe`, 입력 `.dat` 파일은 단일 폴더 내에 위치하는 것을 기준으로 합니다.

## 2. 준비사항

* **Python 3.x**
* **필수 Python 라이브러리:** `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-learn`, `dataframe_image`, `xlsxwriter`
    * (설치 예: `pip install pandas matplotlib scipy scikit-learn dataframe_image xlsxwriter`)
* **GSLIB `gamv.exe` 실행 파일:** 스크립트와 동일 폴더 내 위치
* **입력 데이터 파일:** GSLIB 형식의 `.dat` 파일 (스크립트와 동일 폴더 내 위치)

## 3. 스크립트 기능 및 사용 순서

각 스크립트는 순차적으로 사용하는 것을 권장합니다. 주요 설정값은 각 스크립트 상단 또는 `if __name__ == "__main__":` 블록 내에 정의되어 있습니다.

1.  **`gamv.py` (GSLIB 파라미터 파일 생성)**
    * **기능:** 입력 데이터 파일(`.dat`) 정보 및 사용자 설정(변수, `xlag` 등)을 기반으로 GSLIB `gamv` 프로그램 실행에 필요한 다수의 파라미터 파일(`.par`)을 현재 폴더에 자동 생성합니다.
    * **입력:** `PYTHON_COORD_DATA_FILE`, `GSLIB_TARGET_DATA_FILE` 변수 등에 지정된 `.dat` 파일, 사용자 정의 변수/파라미터.
    * **출력:** `.par` 파일들 (현재 폴더).
![image](https://github.com/user-attachments/assets/e976c9e5-9a76-427a-bef3-512131ef836c)
   사용자가 lag distance를 지정해주면 lag 개수는 자동으로 계산됩니다.
2.  **`gamv2.py` (GSLIB `gamv.exe` 일괄 실행)**
    * **기능:** 현재 폴더 내의 모든 `.par` 파일을 찾아 `gamv.exe`로 순차 실행합니다.
    * **입력:** 현재 폴더 내 `.par` 파일들, `gamv.exe` (현재 폴더).
    * **출력:** `gamv.exe` 실행 결과 파일 (`.out`), 콘솔 로그.

3.  **`gamv3.py` (데이터 기초 분석 및 시각화 - 선택 사항)**
    * **기능:** GSLIB 데이터 파일(`.dat`)의 기초 통계, 공간 분포, 최근린 거리(NND)를 분석하고 시각화합니다. (`xlag` 선정 등에 참고)
    * **입력:** `DATA_FILENAME` 변수에 지정된 `.dat` 파일.
    * **출력:** 분석 플롯 및 통계 테이블/CSV (현재 폴더 내 `preliminary_[데이터파일명]_analysis` 폴더 생성 후 저장).

4.  **`gamv4.py` (실험적 베리오그램 시각화 - 선택 사항)**
    * **기능:** `gamv.exe` 결과 파일(`.out`)을 파싱하여 모든 실험적 베리오그램을 시각화합니다.
    * **입력:** 현재 폴더 내 `.out` 파일들.
    * **출력:** 실험적 베리오그램 플롯 (현재 폴더 내 `variogram_plots` 폴더 생성 후 저장).

5.  **`gamv5.py` (이론적 베리오그램 모델 피팅 - 다중 모델 비교)**
    * **기능:** `.out` 파일의 실험적 베리오그램에 여러 이론 모델(구형, 지수형, 가우시안)을 피팅하고, RMS 기준 최적 모델을 선정하여 결과를 시각화 및 요약합니다.
    * **입력:** 현재 폴더 내 `.out` 파일들.
    * **출력:** 모델 피팅 플롯, 종합 결과 CSV/Excel/표 이미지 (현재 폴더 내 `variogram_analysis` 폴더 생성 후 저장).

6.  **`gamv6.py` (대표 이론 모델 피팅 - 단일 모델 적용)**
    * **기능:** `.out` 파일의 실험적 베리오그램에 gamv5.py분석을 통해 선정된 대표모델을 사용자가 변수/카테고리별로 사전 정의하여 피팅하고 결과를 시각화 및 요약합니다.
    * **입력:** 현재 폴더 내 `.out` 파일들, 스크립트 내 `REPRESENTATIVE_MODELS_CONFIG` 설정.
    * **출력:** 대표 모델 피팅 플롯, 종합 결과 CSV/Excel/표 이미지 (현재 폴더 내 `variogram_analysis_rep_model` 폴더 생성 후 저장).
![image](https://github.com/user-attachments/assets/f13ba2dc-8365-405c-9f20-4b99f1a264af)
    선정된 대표 이론 모델을 수동으로 지정해주었습니다
## 4. 주요 설정

각 스크립트 상단 또는 `if __name__ == "__main__":` 블록 내에 **파일 경로, 분석 변수명, GSLIB 파라미터, 모델 피팅 조건** 등 사용자가 환경에 맞게 수정해야 하는 주요 변수들이 정의되어 있습니다. 실행 전 반드시 확인 및 수정을 진행해 주십시오. (본 README는 모든 파일이 단일 폴더에 있음을 가정하므로, 대부분의 경로 설정은 현재 디렉토리(`.`)를 기본으로 합니다.)

## 5. 주요 출력물

* GSLIB 파라미터 파일 (`.par`)
* GSLIB 실행 결과 파일 (`.out`)
* 데이터 기초 분석 플롯, 통계 CSV 및 표 이미지 (`gamv3.txt` 실행 시)
* 실험적/이론적 베리오그램 플롯 이미지
* 모델 피팅 결과 파라미터 및 통계량을 담은 CSV, Excel, 표 이미지

생성된 결과물들은 스크립트에 지정된 각기 다른 하위 폴더(예: `preliminary...`, `variogram_plots`, `variogram_analysis`, `variogram_analysis_rep_model`)에 체계적으로 저장됩니다.
