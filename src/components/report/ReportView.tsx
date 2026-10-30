import React, { useMemo, useState } from 'react';
import {
  Alert,
  Linking,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  type LayoutChangeEvent,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import type {
  AIReport,
  BodyWatchPoint,
  DiseaseRisk,
  FeatureTrend,
  LifestylePlanItem,
  RiskLevel,
  RiskReportItem,
  SourceRef,
  SupplementCandidate,
} from '../../types/models';

// 리포트 화면 출력 명세(백엔드 back 브랜치 2026-10-28 기준)에 따른 5개 섹션:
// ① 인바디 분석 ② 생활습관 ③ 식습관 ④ 최근 변화 추이 ⑤ 출처

const TEAL = '#00C2A8';
const SOURCES_PREVIEW = 5; // 출처가 많으면 처음엔 이만큼만 보여 준다

const RISK: Record<RiskLevel, { label: string; color: string; bg: string; rank: number }> = {
  low: { label: '낮음', color: '#00897B', bg: '#E0F2F1', rank: 0 },
  medium: { label: '주의', color: '#EF6C00', bg: '#FFF3E0', rank: 1 },
  high: { label: '높음', color: '#C62828', bg: '#FFEBEE', rank: 2 },
};

const CATEGORY: Record<LifestylePlanItem['category'], string> = {
  nutrition: '식사',
  exercise: '운동',
  sleep: '수면',
  habit: '습관',
};

const EVIDENCE: Record<SupplementCandidate['evidence_level'], string> = {
  general: '일반',
  moderate: '보통',
  limited: '제한적',
};

const DIRECTION: Record<FeatureTrend['direction'], string> = {
  up: '증가',
  down: '감소',
  stable: '유지',
};

function sidNumber(sid: string): number {
  const n = Number(sid.replace(/\D/g, ''));
  return Number.isNaN(n) ? 0 : n;
}

function sourceMeta(s: SourceRef): string {
  const parts = [s.publisher, s.year ? String(s.year) : null].filter(Boolean).join(' ');
  const where = s.page ? `p.${s.page}` : s.heading_path ?? '';
  return [parts, where].filter(Boolean).join(' · ');
}

// ---------- 공통 작은 부품 ----------

type SourceMap = Record<string, SourceRef>;

function Citations({ ids, sources }: { ids: string[] | undefined; sources: SourceMap }) {
  const valid = (ids ?? []).filter((id) => sources[id]);
  if (valid.length === 0) return null;

  const open = (s: SourceRef) => {
    const body = [sourceMeta(s), s.heading_path && s.page ? s.heading_path : null]
      .filter(Boolean)
      .join('\n');
    const buttons: { text: string; onPress?: () => void }[] = [{ text: '닫기' }];
    if (s.url) buttons.push({ text: '원문 보기', onPress: () => Linking.openURL(s.url!) });
    Alert.alert(`[${s.sid}] ${s.title}`, body, buttons);
  };

  return (
    <View style={styles.citeRow}>
      {valid.map((id) => (
        <TouchableOpacity
          key={id}
          onPress={() => open(sources[id])}
          hitSlop={{ top: 8, bottom: 8, left: 4, right: 4 }}
          accessibilityLabel={`출처 ${id} 보기`}
        >
          <Text style={styles.cite}>[{id}]</Text>
        </TouchableOpacity>
      ))}
    </View>
  );
}

function Bullet({ text }: { text: string }) {
  return (
    <View style={styles.bulletRow}>
      <Text style={styles.bulletDot}>•</Text>
      <Text style={styles.bulletText}>{text}</Text>
    </View>
  );
}

function Badge({ level }: { level: RiskLevel }) {
  const cfg = RISK[level];
  return (
    <View style={[styles.badge, { backgroundColor: cfg.bg }]}>
      <Text style={[styles.badgeText, { color: cfg.color }]}>{cfg.label}</Text>
    </View>
  );
}

function Section({
  icon,
  title,
  children,
}: {
  icon: keyof typeof Ionicons.glyphMap;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <View style={styles.section}>
      <View style={styles.sectionHeader}>
        <Ionicons name={icon} size={18} color={TEAL} />
        <Text style={styles.sectionTitle}>{title}</Text>
      </View>
      {children}
    </View>
  );
}

// ---------- ① 인바디 분석 ----------

function RiskCard({
  risk,
  factors,
  sources,
  highlighted,
}: {
  risk: RiskReportItem;
  factors: DiseaseRisk['top_factors'];
  sources: SourceMap;
  highlighted: boolean;
}) {
  const cfg = RISK[risk.risk_level];
  const pct = Math.min(Math.max(risk.probability * 100, 0), 100);
  return (
    <View style={[styles.card, highlighted && { borderColor: cfg.color, borderWidth: 1.5 }]}>
      <View style={styles.cardHeader}>
        <View style={styles.titleWithCite}>
          <Text style={styles.cardTitle}>{risk.disease}</Text>
          <Citations ids={risk.source_ids} sources={sources} />
        </View>
        <Badge level={risk.risk_level} />
      </View>
      <View style={styles.probRow}>
        <View style={styles.probBarBg}>
          <View style={[styles.probBarFill, { width: `${pct}%`, backgroundColor: cfg.color }]} />
        </View>
        <Text style={[styles.probPct, { color: cfg.color }]}>{pct.toFixed(1)}%</Text>
      </View>
      <Text style={styles.body}>{risk.message}</Text>
      {risk.main_factors.map((f, i) => (
        <Bullet key={i} text={f} />
      ))}
      {factors.length > 0 && (
        <View style={styles.chipRow}>
          {factors.map((f, i) => (
            <View key={i} style={styles.chip}>
              <Text style={[styles.chipText, { color: f.direction === 'up' ? '#C62828' : '#00897B' }]}>
                {f.feature_kor} {f.direction === 'up' ? '↑' : '↓'}
              </Text>
            </View>
          ))}
        </View>
      )}
    </View>
  );
}

function WatchPoint({ item, sources }: { item: BodyWatchPoint; sources: SourceMap }) {
  return (
    <View style={styles.item}>
      <View style={styles.cardHeader}>
        <View style={styles.titleWithCite}>
          <Text style={styles.itemTitle}>{item.metric}</Text>
          <Citations ids={item.source_ids} sources={sources} />
        </View>
        <Badge level={item.priority} />
      </View>
      <Text style={styles.value}>{item.current_value}</Text>
      <Text style={styles.body}>{item.interpretation}</Text>
    </View>
  );
}

// ---------- ②③ 생활습관·식습관 ----------

function PlanItem({ item, sources }: { item: LifestylePlanItem; sources: SourceMap }) {
  return (
    <View style={styles.item}>
      <View style={styles.titleWithCite}>
        <Text style={styles.itemTitle}>
          <Text style={styles.category}>{CATEGORY[item.category]} · </Text>
          {item.title}
        </Text>
        <Citations ids={item.source_ids} sources={sources} />
      </View>
      <Text style={styles.body}>{item.reason}</Text>
      {item.actions.map((a, i) => (
        <Bullet key={i} text={a} />
      ))}
    </View>
  );
}

function Supplement({ item, sources }: { item: SupplementCandidate; sources: SourceMap }) {
  return (
    <View style={styles.item}>
      <View style={styles.cardHeader}>
        <View style={styles.titleWithCite}>
          <Text style={styles.itemTitle}>{item.name}</Text>
          <Citations ids={item.source_ids} sources={sources} />
        </View>
        <Text style={styles.meta}>근거 {EVIDENCE[item.evidence_level]}</Text>
      </View>
      <Text style={styles.body}>{item.purpose}</Text>
      <Text style={styles.body}>{item.why_consider}</Text>
      {item.cautions.map((c, i) => (
        <Bullet key={i} text={c} />
      ))}
      {item.not_recommended_if.length > 0 && (
        <Text style={styles.avoid}>이런 경우 피하세요: {item.not_recommended_if.join(', ')}</Text>
      )}
    </View>
  );
}

// ---------- ④ 최근 변화 추이 ----------

const CHART_HEIGHT = 70;

// 별도 차트 라이브러리 없이 점과 선(회전한 View)으로 그리는 작은 꺾은선 그래프
function MiniLineChart({ values, color }: { values: number[]; color: string }) {
  const [width, setWidth] = useState(0);
  const onLayout = (e: LayoutChangeEvent) => setWidth(e.nativeEvent.layout.width);

  const points = useMemo(() => {
    if (width === 0 || values.length === 0) return [];
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    const pad = 6;
    const stepX = values.length > 1 ? (width - pad * 2) / (values.length - 1) : 0;
    return values.map((v, i) => ({
      x: pad + i * stepX,
      y: pad + (1 - (v - min) / range) * (CHART_HEIGHT - pad * 2),
    }));
  }, [values, width]);

  return (
    <View style={styles.chart} onLayout={onLayout}>
      {points.slice(1).map((p, i) => {
        const a = points[i];
        const dx = p.x - a.x;
        const dy = p.y - a.y;
        const len = Math.sqrt(dx * dx + dy * dy);
        const angle = (Math.atan2(dy, dx) * 180) / Math.PI;
        return (
          <View
            key={`l${i}`}
            style={[
              styles.chartLine,
              {
                backgroundColor: color,
                width: len,
                left: (a.x + p.x) / 2 - len / 2,
                top: (a.y + p.y) / 2 - 1,
                transform: [{ rotate: `${angle}deg` }],
              },
            ]}
          />
        );
      })}
      {points.map((p, i) => (
        <View
          key={`d${i}`}
          style={[styles.chartDot, { borderColor: color, left: p.x - 4, top: p.y - 4 }]}
        />
      ))}
    </View>
  );
}

function shortDate(d: string): string {
  const date = new Date(d);
  if (Number.isNaN(date.getTime())) return d;
  return `${date.getFullYear() % 100}.${date.getMonth() + 1}`;
}

function TrendCard({ trend }: { trend: FeatureTrend }) {
  const color = trend.is_risk_worsening ? '#C62828' : TEAL;
  const change = `${trend.total_change > 0 ? '+' : ''}${trend.total_change.toFixed(1)}`;
  return (
    <View style={[styles.item, trend.is_risk_worsening && styles.trendWarn]}>
      <View style={styles.cardHeader}>
        <Text style={styles.itemTitle}>{trend.feature_kor}</Text>
        <Text style={[styles.trendChange, { color }]}>
          {DIRECTION[trend.direction]} {change}
        </Text>
      </View>
      {trend.is_risk_worsening && (
        <Text style={styles.trendWarnText}>위험을 높이는 방향으로 변하는 중이에요</Text>
      )}
      <MiniLineChart values={trend.values} color={color} />
      <View style={styles.chartAxis}>
        <Text style={styles.meta}>{shortDate(trend.dates[0])}</Text>
        <Text style={styles.meta}>{shortDate(trend.dates[trend.dates.length - 1])}</Text>
      </View>
    </View>
  );
}

// ---------- ⑤ 출처 ----------

function SourceList({ sources }: { sources: SourceRef[] }) {
  const [expanded, setExpanded] = useState(false);
  const sorted = [...sources].sort((a, b) => sidNumber(a.sid) - sidNumber(b.sid));
  const shown = expanded ? sorted : sorted.slice(0, SOURCES_PREVIEW);

  return (
    <View>
      {shown.map((s) => (
        <TouchableOpacity
          key={s.sid}
          style={styles.sourceRow}
          disabled={!s.url}
          onPress={() => s.url && Linking.openURL(s.url)}
        >
          <Text style={styles.sourceSid}>[{s.sid}]</Text>
          <View style={styles.flex1}>
            <Text style={styles.sourceTitle}>
              {s.title}
              {s.url ? <Text style={styles.sourceLink}>  열기 ↗</Text> : null}
            </Text>
            <Text style={styles.meta}>{sourceMeta(s)}</Text>
          </View>
        </TouchableOpacity>
      ))}
      {sorted.length > SOURCES_PREVIEW && (
        <TouchableOpacity style={styles.moreButton} onPress={() => setExpanded((v) => !v)}>
          <Text style={styles.moreText}>
            {expanded ? '접기' : `출처 ${sorted.length - SOURCES_PREVIEW}개 더 보기`}
          </Text>
        </TouchableOpacity>
      )}
    </View>
  );
}

// ---------- 전체 ----------

export default function ReportView({ report, dateLabel }: { report: AIReport; dateLabel: string }) {
  const sources = report.sources ?? [];
  const sourceMap = useMemo<SourceMap>(
    () => Object.fromEntries(sources.map((s) => [s.sid, s])),
    [sources],
  );

  const factorsOf = (disease: string) =>
    report.disease_risks?.find((d) => d.disease === disease)?.top_factors ?? [];

  // 가장 위험한 질환: 등급이 가장 높고, 같으면 확률이 높은 것
  const top = [...report.risk_report].sort(
    (a, b) => RISK[b.risk_level].rank - RISK[a.risk_level].rank || b.probability - a.probability,
  )[0];

  const body = report.body_composition_report;
  const lifestyle = report.lifestyle_plan.filter((p) => p.category !== 'nutrition');
  const nutrition = report.lifestyle_plan.filter((p) => p.category === 'nutrition');
  const supplements = report.supplement_candidates ?? [];
  const trends = report.trend_analysis?.feature_trends ?? [];

  return (
    <View>
      {/* 요약 */}
      <View style={styles.summary}>
        <Text style={styles.summaryDate}>{dateLabel} 측정 기준</Text>
        <Text style={styles.summaryText}>{report.summary.overall_message}</Text>
        {top && (
          <View style={styles.summaryTop}>
            <Text style={styles.summaryTopLabel}>가장 주의할 질환</Text>
            <Text style={[styles.summaryTopValue, { color: RISK[top.risk_level].color }]}>
              {top.disease} {(top.probability * 100).toFixed(1)}% · {RISK[top.risk_level].label}
            </Text>
          </View>
        )}
      </View>

      {/* ① 인바디 분석 */}
      <Section icon="pulse-outline" title="인바디 분석">
        {report.risk_report.map((r) => (
          <RiskCard
            key={r.disease}
            risk={r}
            factors={factorsOf(r.disease)}
            sources={sourceMap}
            highlighted={top?.disease === r.disease && r.risk_level !== 'low'}
          />
        ))}
        {(body.watch_points.length > 0 || body.strengths.length > 0) && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>관련 체성분</Text>
            {body.watch_points.map((w, i) => (
              <WatchPoint key={`w${i}`} item={w} sources={sourceMap} />
            ))}
            {body.strengths.map((s, i) => (
              <View key={`s${i}`} style={styles.item}>
                <View style={styles.titleWithCite}>
                  <Text style={styles.itemTitle}>{s.metric}</Text>
                  <Text style={styles.good}>잘하고 있어요</Text>
                </View>
                <Text style={styles.body}>{s.message}</Text>
              </View>
            ))}
          </View>
        )}
      </Section>

      {/* ② 생활습관 */}
      {lifestyle.length > 0 && (
        <Section icon="walk-outline" title="생활습관">
          <View style={styles.card}>
            {lifestyle.map((p, i) => (
              <PlanItem key={i} item={p} sources={sourceMap} />
            ))}
          </View>
        </Section>
      )}

      {/* ③ 식습관 */}
      {(nutrition.length > 0 || supplements.length > 0) && (
        <Section icon="restaurant-outline" title="식습관">
          {nutrition.length > 0 && (
            <View style={styles.card}>
              {nutrition.map((p, i) => (
                <PlanItem key={i} item={p} sources={sourceMap} />
              ))}
            </View>
          )}
          {supplements.length > 0 && (
            <View style={styles.card}>
              <Text style={styles.cardTitle}>영양제·보충제</Text>
              {supplements.map((s, i) => (
                <Supplement key={i} item={s} sources={sourceMap} />
              ))}
            </View>
          )}
        </Section>
      )}

      {/* ④ 최근 변화 추이: 데이터가 없으면 섹션 자체를 숨긴다 */}
      {trends.length > 0 && (
        <Section icon="trending-up-outline" title="최근 변화 추이">
          <View style={styles.card}>
            <Text style={styles.meta}>
              기록 {report.trend_analysis!.record_count}개 · {report.trend_analysis!.span_days}일 동안
            </Text>
            {trends.map((t) => (
              <TrendCard key={t.feature_kor} trend={t} />
            ))}
          </View>
        </Section>
      )}

      {/* ⑤ 출처 */}
      <Section icon="library-outline" title="출처">
        <View style={styles.card}>
          {sources.length > 0 ? (
            <SourceList sources={sources} />
          ) : (
            <Text style={styles.meta}>이 리포트에 인용된 문서가 없습니다.</Text>
          )}
          <View style={styles.disclaimer}>
            <Text style={styles.disclaimerText}>{report.cautions.medical_disclaimer}</Text>
            {report.cautions.consult_doctor_if.length > 0 && (
              <>
                <Text style={styles.disclaimerSub}>이런 경우 의사와 상담하세요</Text>
                {report.cautions.consult_doctor_if.map((c, i) => (
                  <Bullet key={i} text={c} />
                ))}
              </>
            )}
          </View>
        </View>
      </Section>
    </View>
  );
}

const styles = StyleSheet.create({
  flex1: { flex: 1 },

  summary: {
    backgroundColor: '#fff',
    borderRadius: 14,
    padding: 16,
    borderLeftWidth: 4,
    borderLeftColor: TEAL,
    gap: 8,
  },
  summaryDate: { fontSize: 12, color: '#999' },
  summaryText: { fontSize: 15, lineHeight: 22, color: '#1A1A1A', fontWeight: '600' },
  summaryTop: { flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 2 },
  summaryTopLabel: { fontSize: 12, color: '#888' },
  summaryTopValue: { fontSize: 13, fontWeight: '700' },

  section: { marginTop: 24, gap: 10 },
  sectionHeader: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  sectionTitle: { fontSize: 17, fontWeight: '700', color: '#1A1A1A' },

  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    borderWidth: 1,
    borderColor: '#F0F0F0',
    gap: 8,
  },
  cardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: 8 },
  cardTitle: { fontSize: 16, fontWeight: '700', color: '#1A1A1A' },
  titleWithCite: { flex: 1, flexDirection: 'row', alignItems: 'center', flexWrap: 'wrap', gap: 4 },

  item: { borderTopWidth: 1, borderTopColor: '#F2F2F2', paddingTop: 12, marginTop: 4, gap: 4 },
  itemTitle: { fontSize: 14, fontWeight: '700', color: '#1A1A1A' },
  category: { color: '#00A38D' },
  value: { fontSize: 14, fontWeight: '600', color: '#333' },
  body: { fontSize: 14, lineHeight: 21, color: '#444' },
  meta: { fontSize: 12, color: '#888', lineHeight: 18 },
  good: { fontSize: 12, fontWeight: '700', color: '#00897B' },
  avoid: { fontSize: 13, color: '#C62828', lineHeight: 19, marginTop: 2 },

  badge: { paddingHorizontal: 10, paddingVertical: 3, borderRadius: 20 },
  badgeText: { fontSize: 12, fontWeight: '700' },

  citeRow: { flexDirection: 'row', gap: 2 },
  cite: { fontSize: 11, fontWeight: '700', color: '#00A38D' },

  probRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  probBarBg: { flex: 1, height: 8, backgroundColor: '#F0F0F0', borderRadius: 4, overflow: 'hidden' },
  probBarFill: { height: '100%', borderRadius: 4 },
  probPct: { fontSize: 14, fontWeight: '700', width: 50, textAlign: 'right' },

  chipRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginTop: 2 },
  chip: { backgroundColor: '#F5F5F5', borderRadius: 12, paddingHorizontal: 8, paddingVertical: 3 },
  chipText: { fontSize: 12, fontWeight: '600' },

  bulletRow: { flexDirection: 'row', gap: 8, marginTop: 2 },
  bulletDot: { color: TEAL, fontSize: 14, lineHeight: 21 },
  bulletText: { flex: 1, fontSize: 14, color: '#444', lineHeight: 21 },

  chart: { height: CHART_HEIGHT, marginTop: 6, backgroundColor: '#FAFAFA', borderRadius: 8 },
  chartLine: { position: 'absolute', height: 2 },
  chartDot: {
    position: 'absolute',
    width: 8,
    height: 8,
    borderRadius: 4,
    borderWidth: 2,
    backgroundColor: '#fff',
  },
  chartAxis: { flexDirection: 'row', justifyContent: 'space-between' },
  trendChange: { fontSize: 13, fontWeight: '700' },
  trendWarn: { backgroundColor: '#FFF8F8', borderRadius: 8, paddingHorizontal: 8, paddingBottom: 8 },
  trendWarnText: { fontSize: 12, color: '#C62828' },

  sourceRow: { flexDirection: 'row', gap: 8, paddingVertical: 8, borderBottomWidth: 1, borderBottomColor: '#F2F2F2' },
  sourceSid: { fontSize: 12, fontWeight: '700', color: '#00A38D', width: 34 },
  sourceTitle: { fontSize: 13, color: '#1A1A1A', lineHeight: 19 },
  sourceLink: { fontSize: 12, color: '#00A38D', fontWeight: '600' },
  moreButton: { paddingVertical: 10, alignItems: 'center' },
  moreText: { fontSize: 13, fontWeight: '600', color: '#00A38D' },

  disclaimer: { marginTop: 8, backgroundColor: '#FFF8E1', borderRadius: 10, padding: 12, gap: 4 },
  disclaimerText: { fontSize: 13, lineHeight: 20, color: '#92400E' },
  disclaimerSub: { fontSize: 12, fontWeight: '700', color: '#92400E', marginTop: 4 },
});
