const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

async function downloadAllPDFs() {
  // 전체 페이지 수 계산 (1147개 / 10 = 115페이지)
  const totalPages = 115;
  const startPage = 1; // 1페이지부터 재시작하여 누락 파일 다운로드
  let downloadedCount = 0;
  let skippedCount = 0;
  let errorCount = 0;
  let consecutiveErrors = 0;

  // User-Agent 리스트 (랜덤 로테이션)
  const userAgents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14.1; rv:120.0) Gecko/20100101 Firefox/120.0'
  ];

  let browser = await chromium.launch({ headless: true });
  let context = await browser.newContext({
    acceptDownloads: true,
    userAgent: userAgents[Math.floor(Math.random() * userAgents.length)],
    viewport: { width: 1920, height: 1080 },
    locale: 'ko-KR',
    timezoneId: 'Asia/Seoul'
  });
  let page = await context.newPage();

  for (let pageNum = startPage; pageNum <= totalPages; pageNum++) {
    console.log(`\n페이지 ${pageNum}/${totalPages} 처리 중... (에러: ${errorCount}, 연속: ${consecutiveErrors})`);

    // 연속 에러가 5번 이상이면 브라우저 재시작 + 긴 대기
    if (consecutiveErrors >= 5) {
      console.log(`\n⚠️  연속 에러 ${consecutiveErrors}번 발생, 브라우저 재시작 및 대기 중...`);
      await page.close().catch(() => {});
      await context.close().catch(() => {});
      await browser.close().catch(() => {});

      // 30-60초 랜덤 대기
      const waitTime = 30000 + Math.random() * 30000;
      console.log(`   ${Math.round(waitTime/1000)}초 대기 중...`);
      await new Promise(resolve => setTimeout(resolve, waitTime));

      // 새 브라우저 인스턴스 시작
      browser = await chromium.launch({ headless: true });
      context = await browser.newContext({
        acceptDownloads: true,
        userAgent: userAgents[Math.floor(Math.random() * userAgents.length)],
        viewport: {
          width: 1366 + Math.floor(Math.random() * 500),
          height: 768 + Math.floor(Math.random() * 300)
        },
        locale: 'ko-KR',
        timezoneId: 'Asia/Seoul'
      });
      page = await context.newPage();
      consecutiveErrors = 0;
      console.log(`   브라우저 재시작 완료, 계속 진행...`);
    }

    try {
      // 페이지로 직접 이동
      const url = `https://board.kra.co.kr/board/viewBoard.do?boardNo=211&strToken=&usernm=&currentPage=${pageNum}`;
      await page.goto(url, { waitUntil: 'networkidle', timeout: 30000 });

      // 랜덤 대기 (2-5초)
      const randomWait = 2000 + Math.random() * 3000;
      await page.waitForTimeout(randomWait);

      // 현재 페이지의 모든 게시물 행 가져오기
      const rows = await page.locator('table tbody tr').all();
      consecutiveErrors = 0; // 페이지 접속 성공

      for (let i = 0; i < rows.length; i++) {
        try {
          const row = rows[i];

          // 제목 셀에서 날짜 추출
          const titleText = await row.locator('td').nth(1).textContent({ timeout: 5000 });
          if (!titleText || !titleText.includes('경주성적표')) {
            continue;
          }

          // 날짜 추출 (예: "2025.09.28.(일) 경주성적표" 또는 "2023. 2. 12.(일) 경주성적표")
          let dateMatch = titleText.match(/(\d{4})\.(\d{2})\.(\d{2})/);
          if (!dateMatch) {
            // 공백이 있는 형식 시도
            dateMatch = titleText.match(/(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})/);
          }
          if (!dateMatch) {
            console.log(`  날짜 추출 실패: ${titleText}`);
            continue;
          }

          const year = dateMatch[1];
          const month = dateMatch[2].padStart(2, '0');
          const day = dateMatch[3].padStart(2, '0');
          const fileName = `${year}-${month}-${day}.pdf`;
          const filePath = path.join('/Users/ryan/horse_park', fileName);

          // 이미 다운로드된 파일인지 확인
          if (fs.existsSync(filePath)) {
            console.log(`  ✓ 이미 존재함: ${fileName}`);
            skippedCount++;
            continue;
          }

          // 게시물 클릭
          await row.locator('td').nth(1).locator('a').click({ timeout: 10000 });
          await page.waitForTimeout(1000 + Math.random() * 1000);

          // 다운로드 링크 클릭 및 파일 저장
          const downloadPromise = page.waitForEvent('download', { timeout: 15000 });
          await page.locator('a:has-text("다운로드")').click({ timeout: 10000 });
          const download = await downloadPromise;

          // 다운로드된 파일을 올바른 이름으로 저장
          await download.saveAs(filePath);
          downloadedCount++;
          console.log(`  ✓ 다운로드 완료: ${fileName} (${downloadedCount}/${1147})`);

          // 목록으로 돌아가기 - URL로 직접 이동
          const listUrl = `https://board.kra.co.kr/board/viewBoard.do?boardNo=211&strToken=&usernm=&currentPage=${pageNum}`;
          await page.goto(listUrl, { waitUntil: 'networkidle', timeout: 30000 });
          await page.waitForTimeout(1000 + Math.random() * 1000);

        } catch (error) {
          console.log(`  ✗ 오류 발생: ${error.message}`);
          errorCount++;
          // 오류 발생 시 목록 페이지로 돌아가기
          try {
            const listUrl = `https://board.kra.co.kr/board/viewBoard.do?boardNo=211&strToken=&usernm=&currentPage=${pageNum}`;
            await page.goto(listUrl, { waitUntil: 'networkidle', timeout: 30000 });
            await page.waitForTimeout(1000);
          } catch (e) {
            console.log(`  ✗ 복구 실패: ${e.message}`);
          }
        }
      }
    } catch (error) {
      console.log(`페이지 ${pageNum} 처리 중 오류: ${error.message}`);
      errorCount++;
      consecutiveErrors++;

      // ERR_EMPTY_RESPONSE 발생시 추가 대기
      if (error.message.includes('ERR_EMPTY_RESPONSE')) {
        const waitTime = 10000 + Math.random() * 10000;
        console.log(`  차단 감지, ${Math.round(waitTime/1000)}초 대기...`);
        await new Promise(resolve => setTimeout(resolve, waitTime));
      }
    }
  }

  console.log(`\n\n다운로드 완료!`);
  console.log(`- 새로 다운로드: ${downloadedCount}개`);
  console.log(`- 이미 존재함: ${skippedCount}개`);
  console.log(`- 총: ${downloadedCount + skippedCount}개`);

  await browser.close();
}

downloadAllPDFs().catch(console.error);