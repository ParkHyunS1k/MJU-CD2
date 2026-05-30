import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import { api } from '../lib/api';
import type {
  AIReport,
  BodyWatchPoint,
  InBodyRecord,
  LifestylePlanItem,
  RiskLevel,
  RiskReportItem,
  SupplementCandidate,
} from '../types/models';

const TEAL = '#00C2A8';

const DISEASE_KOR: Record<string, string> = {
  hypertension: '고혈압',
  diabetes: '당뇨병',
  cholesterol: '고콜레스테롤',
  triglyceride: '고중성지방',
};

const CATEGORY_KOR: Record<LifestylePlanItem['category'], string> = {
  nutrition: '식사',
  exercise: '운동',
  sleep: '수면',
  habit: '습관',
};

const EVIDENCE_KOR: Record<SupplementCandidate['evidence_level'], string> = {
  general: '일반',
  moderate: '보통',
  limited: '제한적',
};

const RISK_CONFIG: Record<RiskLevel, { label: string; color: string; bg: string }> = {
  low:    { label: '낮음', color: '#009688', bg: '#E0F2F1' },
  medium: { label: '주의', color: '#FF8F00', bg: '#FFF8E1' },
  high:   { label: '높음', color: '#C62828', bg: '#FFEBEE' },
};

function formatDate(dateStr: string | null): string {
  if (!dateStr) return '-';
  const d = new Date(dateStr);
  return `${d.getFullYear()}.${d.getMonth() + 1}.${d.getDate()}`;
}

function SectionCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <View style={styles.sectionCard}>
      <Text style={styles.sectionCardTitle}>{title}</Text>
      {children}
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

function RiskCard({ risk }: { risk: RiskReportItem }) {
  const cfg = RISK_CONFIG[risk.risk_level];
  const pct = (risk.probability * 100).toFixed(1);
  return (
    <View style={styles.riskCard}>
      <View style={styles.riskCardHeader}>
        <Text style={styles.riskCardName}>{DISEASE_KOR[risk.disease] ?? risk.disease}</Text>
        <View style={[styles.riskBadge, { backgroundColor: cfg.bg }]}>
          <Text style={[styles.riskBadgeText, { color: cfg.color }]}>{cfg.label}</Text>
        </View>
      </View>
      <View style={styles.probRow}>
        <View style={styles.probBarBg}>
          <View style={[styles.probBarFill, { width: `${risk.probability * 100}%` as any, backgroundColor: cfg.color }]} />
        </View>
        <Text style={[styles.probPct, { color: cfg.color }]}>{pct}%</Text>
      </View>
      <Text style={styles.riskMessage}>{risk.message}</Text>
      {risk.main_factors.map((f, i) => <Bullet key={i} text={f} />)}
    </View>
  );
}

function WatchPointCard({ item }: { item: BodyWatchPoint }) {
  const cfg = RISK_CONFIG[item.priority];
  return (
    <View style={styles.detailBlock}>
      <View style={styles.detailHeader}>
        <Text style={styles.detailTitle}>{item.metric}</Text>
        <View style={[styles.smallBadge, { backgroundColor: cfg.bg }]}>
          <Text style={[styles.smallBadgeText, { color: cfg.color }]}>{cfg.label}</Text>
        </View>
      </View>
      <Text style={styles.detailBody}>{item.current_value}</Text>
      <Text style={styles.detailBody}>{item.interpretation}</Text>
    </View>
  );
}

function PlanCard({ item }: { item: LifestylePlanItem }) {
  return (
    <View style={styles.detailBlock}>
      <Text style={styles.detailTitle}>
        {CATEGORY_KOR[item.category]} · {item.title}
      </Text>
      <Text style={styles.detailBody}>{item.reason}</Text>
      {item.actions.map((a, i) => <Bullet key={i} text={a} />)}
    </View>
  );
}

function SupplementCard({ item }: { item: SupplementCandidate }) {
  return (
    <View style={styles.detailBlock}>
      <View style={styles.detailHeader}>
        <Text style={styles.detailTitle}>{item.name}</Text>
        <Text style={styles.metaText}>근거 {EVIDENCE_KOR[item.evidence_level]}</Text>
      </View>
      <Text style={styles.detailBody}>{item.purpose}</Text>
      <Text style={styles.detailBody}>{item.why_consider}</Text>
      <Text style={styles.metaText}>관련 지표: {item.related_metrics.join(', ')}</Text>
      {item.cautions.map((c, i) => <Bullet key={i} text={c} />)}
      {item.not_recommended_if.length > 0 && (
        <Text style={styles.metaText}>피해야 할 경우: {item.not_recommended_if.join(', ')}</Text>
      )}
    </View>
  );
}

function ReportBody({ report, dateStr }: { report: AIReport; dateStr: string }) {
  return (
    <>
      {/* 종합 건강 요약 */}
      <View style={styles.summaryCard}>
        <Text style={styles.summaryCardTitle}>종합 건강 요약</Text>
        <View style={styles.summaryGrid}>
          <View style={styles.summaryCol}>
            <Text style={styles.summaryLabel}>진반적 위험도</Text>
            <Text style={styles.summaryValue}>
              {report.risk_report.length > 0 ? RISK_CONFIG[report.risk_report[0].risk_level].label : '-'}
            </Text>
          </View>
          <View style={styles.summaryCol}>
            <Text style={styles.summaryLabel}>분석 일정</Text>
            <Text style={styles.summaryValue}>{dateStr}</Text>
          </View>
        </View>
        <Text style={styles.summaryBig}>{report.risk_report.length}개 지표</Text>
        <Text style={styles.summaryNote}>{report.summary.overall_message}</Text>
      </View>

      {/* 세부 위험도 분석 */}
      {report.risk_report.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>세부 위험도 분석</Text>
          {report.risk_report.map((r) => <RiskCard key={r.disease} risk={r} />)}
        </View>
      )}

      {/* 인바디 해석 */}
      {(report.body_composition_report.strengths.length > 0 ||
        report.body_composition_report.watch_points.length > 0) && (
        <SectionCard title="인바디 해석">
          {report.body_composition_report.strengths.map((s, i) => (
            <View key={i} style={styles.detailBlock}>
              <Text style={styles.detailTitle}>{s.metric}</Text>
              <Text style={styles.detailBody}>{s.message}</Text>
            </View>
          ))}
          {report.body_composition_report.watch_points.map((wp, i) => (
            <WatchPointCard key={i} item={wp} />
          ))}
        </SectionCard>
      )}

      {/* 생활습관 계획 */}
      {report.lifestyle_plan.length > 0 && (
        <SectionCard title="맞춤형 개선 가이드">
          {report.lifestyle_plan.map((item, i) => (
            <PlanCard key={i} item={item} />
          ))}
        </SectionCard>
      )}

      {/* 보충제 후보 */}
      {report.supplement_candidates.length > 0 && (
        <SectionCard title="보충제 후보">
          {report.supplement_candidates.map((item, i) => (
            <SupplementCard key={i} item={item} />
          ))}
        </SectionCard>
      )}

      {/* 주의사항 */}
      <SectionCard title="주의사항">
        <Text style={styles.cautionText}>{report.cautions.medical_disclaimer}</Text>
        {report.cautions.consult_doctor_if.map((item, i) => (
          <Bullet key={i} text={item} />
        ))}
      </SectionCard>

      {/* 액션 버튼 */}
      <View style={styles.actionRow}>
        <TouchableOpacity style={styles.actionBtn}>
          <Ionicons name="download-outline" size={17} color="#333" />
          <Text style={styles.actionBtnText}>리포트 다운로드</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.actionBtn}>
          <Ionicons name="share-outline" size={17} color="#333" />
          <Text style={styles.actionBtnText}>리포트 공유</Text>
        </TouchableOpacity>
      </View>
    </>
  );
}

export default function ReportTabScreen() {
  const [record, setRecord] = useState<InBodyRecord | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.listInBody(1)
      .then((data) => setRecord(data[0] ?? null))
      .catch((e) => Alert.alert('오류', e instanceof Error ? e.message : '불러오기 실패'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color={TEAL} />
      </View>
    );
  }

  if (!record?.ai_report) {
    return (
      <View style={styles.emptyContainer}>
        <View style={styles.header}>
          <Text style={styles.headerTitle}>건강 리포트</Text>
        </View>
        <View style={styles.emptyBody}>
          <Ionicons name="document-text-outline" size={64} color="#CCCCCC" />
          <Text style={styles.emptyTitle}>리포트가 없습니다</Text>
          <Text style={styles.emptyDesc}>
            인바디 파일을 업로드하고{'\n'}AI 분석을 시작하세요.
          </Text>
        </View>
      </View>
    );
  }

  const dateStr = formatDate(record.measured_at);

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.content}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>건강 리포트</Text>
      </View>
      <View style={styles.reportMeta}>
        <Text style={styles.reportTitle}>건강 리포트</Text>
        <Text style={styles.reportDate}>{dateStr} 분석 결과</Text>
      </View>
      <ReportBody report={record.ai_report} dateStr={dateStr} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: '#F5F5F5' },
  content: { paddingBottom: 40 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  emptyContainer: { flex: 1, backgroundColor: '#F5F5F5' },
  emptyBody: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 14, padding: 40 },
  emptyTitle: { fontSize: 18, fontWeight: '700', color: '#333' },
  emptyDesc: { fontSize: 14, color: '#888', textAlign: 'center', lineHeight: 22 },

  header: {
    backgroundColor: TEAL,
    paddingTop: 60,
    paddingBottom: 24,
    paddingHorizontal: 20,
    borderBottomLeftRadius: 24,
    borderBottomRightRadius: 24,
    alignItems: 'center',
  },
  headerTitle: { color: '#fff', fontSize: 18, fontWeight: '700' },

  reportMeta: { paddingHorizontal: 20, paddingTop: 20 },
  reportTitle: { fontSize: 22, fontWeight: '700', color: '#1A1A1A', marginBottom: 4 },
  reportDate: { fontSize: 13, color: '#888', marginBottom: 4 },

  summaryCard: {
    backgroundColor: '#fff',
    borderRadius: 14,
    padding: 16,
    marginHorizontal: 16,
    marginTop: 16,
    borderWidth: 1,
    borderColor: '#F0F0F0',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  summaryCardTitle: { fontSize: 15, fontWeight: '700', color: '#1A1A1A', marginBottom: 14 },
  summaryGrid: { flexDirection: 'row', marginBottom: 14 },
  summaryCol: { flex: 1 },
  summaryLabel: { fontSize: 11, color: '#999', marginBottom: 4 },
  summaryValue: { fontSize: 13, fontWeight: '600', color: '#333' },
  summaryBig: { fontSize: 26, fontWeight: '800', color: '#1A1A1A', marginBottom: 8 },
  summaryNote: { fontSize: 13, color: '#555', lineHeight: 20 },

  section: { marginHorizontal: 16, marginTop: 20 },
  sectionTitle: { fontSize: 17, fontWeight: '700', color: '#1A1A1A', marginBottom: 12 },

  riskCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#F0F0F0',
    gap: 8,
  },
  riskCardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  riskCardName: { fontSize: 16, fontWeight: '700', color: '#1A1A1A' },
  riskBadge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 20 },
  riskBadgeText: { fontSize: 13, fontWeight: '700' },
  probRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  probBarBg: { flex: 1, height: 8, backgroundColor: '#F0F0F0', borderRadius: 4, overflow: 'hidden' },
  probBarFill: { height: '100%', borderRadius: 4 },
  probPct: { fontSize: 14, fontWeight: '700', width: 44, textAlign: 'right' },
  riskMessage: { fontSize: 13, color: '#555', lineHeight: 20 },

  sectionCard: {
    backgroundColor: '#fff',
    borderRadius: 14,
    padding: 16,
    marginHorizontal: 16,
    marginTop: 20,
    borderWidth: 1,
    borderColor: '#F0F0F0',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
    gap: 4,
  },
  sectionCardTitle: { fontSize: 16, fontWeight: '700', color: '#1A1A1A', marginBottom: 8 },

  detailBlock: {
    borderTopWidth: 1,
    borderTopColor: '#F0F0F0',
    paddingTop: 12,
    marginTop: 8,
    gap: 4,
  },
  detailHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: 8 },
  detailTitle: { flex: 1, fontSize: 14, fontWeight: '700', color: '#1A1A1A' },
  detailBody: { fontSize: 14, color: '#444', lineHeight: 21 },
  metaText: { fontSize: 12, color: '#888', lineHeight: 18 },
  smallBadge: { paddingHorizontal: 8, paddingVertical: 3, borderRadius: 12 },
  smallBadgeText: { fontSize: 12, fontWeight: '700' },

  cautionText: { fontSize: 14, color: '#92400E', lineHeight: 21 },

  bulletRow: { flexDirection: 'row', gap: 8, marginTop: 3 },
  bulletDot: { color: TEAL, fontSize: 14, lineHeight: 21 },
  bulletText: { flex: 1, fontSize: 14, color: '#444', lineHeight: 21 },

  actionRow: { flexDirection: 'row', gap: 10, marginHorizontal: 16, marginTop: 20 },
  actionBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    backgroundColor: '#fff',
    borderRadius: 12,
    paddingVertical: 14,
    borderWidth: 1,
    borderColor: '#E5E5E5',
  },
  actionBtnText: { fontSize: 13, color: '#333', fontWeight: '600' },
});
