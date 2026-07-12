| 사용자 행동 데이터 3차원 시각화 대시보드
 다차원 사용자 행동 로그 데이터를 분석 및 전처리하여, 3차원 공간 상에 직관적으로 시각화하는 대시보드로서, 마우스 조작을 통해 정상 유저 군집과 이상 패턴을 실시간으로 탐색할 수 있다.

| 서비스 접속 안내 및 고가용성 정책
 대시보드 가동은 메인 서버로 Streamlit Community Cloud를 사용하고 있다.
단, 무료 클라우드 서버 특성상 일정 기간 외부 접속이 없을 경우 서버가 자동으로 휴면 상태로 전환될 수 있음에 주의.

메인 링크 접속이 불가할 경우, 아래의 가이드에 따라 로컬 환경에서 직접 구동할 수도 있다.

| 메인 서비스 접속 링크: https://sijiro-mhc8nhmtpkslflvqonvcmv.streamlit.app/

| 로컬 실행 가이드 
 만약 웹 배포 환경 접속이 불가능하거나, 직접 코드를 구동해보고 싶은 경우 아래의 절차를 따라 로컬 환경에서 즉시 실행할 수 있다. (사전에 Python 3.8 이상이 필요)

| 저장소 복제 
 git clone https://github.com/sharpboon/Sijiro
 cd Sijiro

| 필수 라이브러리 설치 
 pip install -r requirements.txt

| 애플리케이션 실행 
 streamlit run app.py

※ 명령어 실행 후 브라우저에서 자동으로 localhost 창이 열리며 대시보드가 정상 구동된다.

| 기술 스택 
 Language: Python
 Data Processing: Pandas
 Visualization: Plotly 
 Framework & Deployment: Streamlit, GitHub

