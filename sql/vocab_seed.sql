-- ============================================================================
-- 단어 사전 초기 데이터 — 우송대 성과평가보고서에서 추출·검수한 12개
--
-- 스키마가 아니라 문서에서 나온 데이터다. 다른 문서를 넣으면 여기도 달라진다.
--
-- 추출: LLM 이 부모 청크 단위로 17개를 뽑았고, 그중 5개를 버렸다.
--   AI / DS / XR      일반 약어. 모델이 이미 알고, 문서 고유가 아니다.
--   Agile / SMART     확장어가 아니라 정의 문장이 잡혔다
--                     ('Specific: 구체성, Measurable: 측정가능성, ...')
--   IR성과관리팀        IR 의 확장어가 아니라 IR 을 포함한 조직명이라 짝이 틀렸다
--
-- 아래 12개 중 6개(CEFR, IPA, K-MOOC, T.A.G, WLP, WTP)는 이 문서에서 효과가 없다.
-- 문서가 항상 'WLP(Woosong Learning Program)' 식으로 두 표기를 같이 적어서
-- sparse 가 이미 어느 쪽 질의든 잡는다(실측: 청크 374개 전수 확인). 확장을 해도
-- 나빠지지는 않아서(악화 0건) 다른 문서를 위해 남겨둔다.
-- ============================================================================

-- 축약어
INSERT INTO vocab_short (term) VALUES
    ('CEFR'),
    ('E-PAMS'),
    ('IPA'),
    ('IR'),
    ('K-MOOC'),
    ('MD'),
    ('PAMS'),
    ('Pre-College'),
    ('SolDream+'),
    ('T.A.G'),
    ('WLP'),
    ('WTP')
ON CONFLICT (term) DO NOTHING;


-- 확장어
INSERT INTO vocab_expansion (short_id, term)
SELECT s.id, pair.expansion
FROM (VALUES
    ('CEFR',        'Common European Framework of Reference of Language'),
    ('E-PAMS',      '글로벌공동교육과정'),
    ('IPA',         'Importance-Performance Analysis'),
    -- 공백 변형 둘 다 넣는다. 문서에 두 표기가 다 나온다.
    ('IR',          '대학성과통합관리 시스템'),
    ('IR',          '대학성과통합관리시스템'),
    ('K-MOOC',      '한국형 온라인 공개강좌'),
    ('MD',          '마이크로디그리'),
    -- Asia / Asian 오타 변형. 서로 확장해주면 한쪽으로 물어도 잡힌다(실측 0->4건).
    ('PAMS',        'Partnership of Asia Management Schools'),
    ('PAMS',        'Partnership of Asian Management Schools'),
    ('Pre-College', '우송형 선이수프로그램'),
    ('SolDream+',   '학생역량통합관리시스템'),
    ('T.A.G',       'Technology & Trend Agility Globalization'),
    ('WLP',         'Woosong Learning Program'),
    ('WTP',         'Woosong Teaching Program')
) AS pair(short, expansion)
JOIN vocab_short s ON s.term = pair.short
ON CONFLICT (short_id, term) DO NOTHING;


-- ── 확인 ─────────────────────────────────────────────────────────────────────
SELECT
    (SELECT count(*) FROM vocab_short)     AS 축약어,
    (SELECT count(*) FROM vocab_expansion) AS 확장어;

-- 축약어별로 어떤 확장어가 붙었는지
SELECT s.term AS 축약어, string_agg(e.term, ' | ' ORDER BY e.term) AS 확장어
FROM vocab_short s
LEFT JOIN vocab_expansion e ON e.short_id = s.id
GROUP BY s.term
ORDER BY s.term;
