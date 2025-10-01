const fs = require('fs');

// 다운로드된 파일 목록 가져오기
const files = fs.readdirSync('/Users/ryan/horse_park')
  .filter(f => f.endsWith('.pdf'))
  .map(f => f.replace('.pdf', ''))
  .sort();

console.log(`총 다운로드된 파일: ${files.length}개\n`);

// 날짜별로 그룹화하여 누락 확인
const dateSet = new Set(files);

// 2013-01-01부터 2025-09-30까지 모든 날짜 생성 (토, 일만 - 경마는 주말에만)
const startDate = new Date('2013-01-01');
const endDate = new Date('2025-09-30');
const missing = [];

for (let d = new Date(startDate); d <= endDate; d.setDate(d.getDate() + 1)) {
  const dayOfWeek = d.getDay();
  // 토요일(6) 또는 일요일(0)만 체크
  if (dayOfWeek === 0 || dayOfWeek === 6) {
    const dateStr = d.toISOString().split('T')[0];
    if (!dateSet.has(dateStr)) {
      missing.push(dateStr);
    }
  }
}

console.log(`예상 경마일 중 누락된 날짜: ${missing.length}개\n`);
console.log('누락된 날짜 목록 (최근 50개):');
missing.slice(-50).forEach(date => console.log(date));

// 파일로 저장
fs.writeFileSync('/Users/ryan/horse_park/missing_dates.txt', missing.join('\n'));
console.log(`\n전체 누락 목록이 missing_dates.txt에 저장되었습니다.`);