import React, { useCallback, useMemo, useState } from 'react';
import { RouteProp, useFocusEffect, useRoute } from '@react-navigation/native';
import {
  ActivityIndicator,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import { MiniLineChart, RISK } from '../components/report/ReportView';
import { api } from '../lib/api';
import type { InBodyStackParamList } from '../navigation/types';
import type { InBodyRecord, RiskLevel } from '../types/models';

type Rt = RouteProp<InBodyStackParamList, 'RiskScreen'>;

const TEAL = '#00C2A8';

// 한 질환의 기록별 위험도 (서버 /inbody/risk-trend 에서 뽑아낸 것)
interface Point {
  inbodyId: string;
  date: string;
  pct: number;
  level: RiskLevel;
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return `${d.getFullYear()}.${d.getMonth() + 1}.${d.getDate()}`;
}

function Badge({ level }: { level: RiskLevel }) {
  const cfg = RISK[level];
  return (
    <View style={[styles.badge, { backgroundColor: cfg.bg }]}>
      <Text style={[styles.badgeText, { color: cfg.color }]}>{cfg.label}</Text>
    </View>
  );
}

// 리포트의 질환 카드에서 들어오는 위험도 상세 화면.
// 지금 위험도, 직전 측정 대비 변화, 기록별 위험도 추이 그래프를 보여 준다.
export default function RiskScreen() {
  const { id, disease } = useRoute<Rt>().params;

  const [record, setRecord] = useState<InBodyRecord | null>(null);
  const [points, setPoints] = useState<Point[] | null>(null);
  const [trendError, setTrendError] = useState(false);
  const [loading, setLoading] = useState(true);

  useFocusEffect(
    useCallback(() => {
      let active = true;
      Promise.allSettled([api.getInBody(id), api.getRiskTrend()]).then(([rec, trend]) => {
        if (!active) return;
        if (rec.status === 'fulfilled') setRecord(rec.value);
        if (trend.status === 'fulfilled') {
          setPoints(
            trend.value.flatMap((p) => {
              const r = p.risks.find((x) => x.disease === disease);
              return r
                ? [{ inbodyId: p.inbody_id, date: p.measured_at, pct: r.probability * 100, level: r.risk_level }]
                : [];
            }),
          );
          setTrendError(false);
        } else {
          // 예전 서버에는 /inbody/risk-trend 가 없다
          setTrendError(true);
        }
        setLoading(false);
      });
      return () => {
        active = false;
      };
    }, [id, disease]),
  );

  const report = record?.ai_report;
  const riskItem = report?.risk_report.find((r) => r.disease === disease);
  const factors = report?.disease_risks?.find((d) => d.disease === disease)?.top_factors ?? [];

  // 지금 보고 있는 기록의 위치와 바로 이전 측정
  const currentIndex = points ? points.findIndex((p) => p.inbodyId === id) : -1;
  const current = points && currentIndex >= 0 ? points[currentIndex] : null;
  const previous = points && currentIndex > 0 ? points[currentIndex - 1] : null;
  const change = current && previous ? current.pct - previous.pct : null;

  // 리포트에 있는 값을 우선 쓰고, 없으면 추이 데이터의 값을 쓴다
  const level: RiskLevel | undefined = riskItem?.risk_level ?? current?.level;
  const pct = riskItem ? riskItem.probability * 100 : current?.pct;
  const color = level ? RISK[level].color : TEAL;

  const values = useMemo(() => (points ?? []).map((p) => p.pct), [points]);
  // 세로축을 0%부터 그려서 0.1%p 같은 작은 차이가 큰 변화처럼 보이지 않게 한다.
  // 위쪽은 최댓값보다 넉넉하게, 최소 10%까지.
  const yMax = useMemo(() => {
    const top = Math.max(10, ...values) * 1.2;
    return Math.ceil(top / 5) * 5;
  }, [values]);

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color={TEAL} />
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.scroll}
      contentContainerStyle={styles.content}
      contentInsetAdjustmentBehavior="automatic"
    >
      <Text style={styles.title}>{disease} 위험도</Text>
      <Text style={styles.desc}>
        체성분과 건강 설문으로 추정한 값이에요. 질병의 진단이 아닌 참고용이에요.
      </Text>

      {/* 지금 위험도 */}
      <View style={styles.card}>
        <Text style={styles.cardLabel}>
          {record?.measured_at ? `${formatDate(record.measured_at)} 측정` : '이번 측정'}
        </Text>
        <View style={styles.currentRow}>
          {pct !== undefined && (
            <Text style={[styles.bigPct, { color }]}>{pct.toFixed(1)}%</Text>
          )}
          {level && <Badge level={level} />}
        </View>
        {change !== null && previous && (
          <View style={styles.changeRow}>
            <Ionicons
              name={Math.abs(change) < 0.05 ? 'remove' : change > 0 ? 'arrow-up' : 'arrow-down'}
              size={14}
              color={Math.abs(change) < 0.05 ? '#888' : change > 0 ? '#C62828' : '#00897B'}
            />
            <Text style={styles.changeText}>
              지난 측정({formatDate(previous.date)})보다{' '}
              {Math.abs(change) < 0.05
                ? '변화 없음'
                : `${Math.abs(change).toFixed(1)}%p ${change > 0 ? '높아졌어요' : '낮아졌어요'}`}
            </Text>
          </View>
        )}
        {riskItem?.message ? <Text style={styles.body}>{riskItem.message}</Text> : null}
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

      {/* 위험도 변화 추이 */}
      <Text style={styles.subTitle}>위험도 변화 추이</Text>
      <View style={styles.card}>
        {trendError ? (
          <Text style={styles.empty}>
            위험도 추이를 불러오지 못했어요. 서버가 최신 버전인지 확인해 주세요.
          </Text>
        ) : !points || points.length < 2 ? (
          <Text style={styles.empty}>
            측정 기록이 2개 이상 쌓이면 위험도가 어떻게 바뀌었는지 그래프로 보여 드려요.
          </Text>
        ) : (
          <>
            <Text style={styles.cardLabel}>
              측정 {points.length}회 · 속을 채운 점이 지금 보고 있는 측정
            </Text>
            <View style={styles.chartWrap}>
              <View style={styles.yAxis}>
                <Text style={styles.axisText}>{yMax}%</Text>
                <Text style={styles.axisText}>0%</Text>
              </View>
              <View style={styles.flex1}>
                <MiniLineChart
                  values={values}
                  color={color}
                  height={120}
                  highlightIndex={currentIndex}
                  yMin={0}
                  yMax={yMax}
                />
              </View>
            </View>
            <View style={styles.axis}>
              <Text style={styles.axisText}>{formatDate(points[0].date)}</Text>
              <Text style={styles.axisText}>{formatDate(points[points.length - 1].date)}</Text>
            </View>

            {/* 측정별 값 (최근 것이 위) */}
            {[...points].reverse().map((p) => (
              <View key={p.inbodyId} style={[styles.pointRow, p.inbodyId === id && styles.pointRowCurrent]}>
                <Text style={styles.pointDate}>{formatDate(p.date)}</Text>
                <Text style={[styles.pointPct, { color: RISK[p.level].color }]}>{p.pct.toFixed(1)}%</Text>
                <Badge level={p.level} />
              </View>
            ))}
          </>
        )}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: '#F5F5F5' },
  content: { padding: 20, paddingBottom: 40 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#F5F5F5' },

  title: { fontSize: 22, fontWeight: '700', color: '#1A1A1A', marginBottom: 6 },
  desc: { fontSize: 13, color: '#888', lineHeight: 19, marginBottom: 16 },
  subTitle: { fontSize: 16, fontWeight: '700', color: '#1A1A1A', marginTop: 8, marginBottom: 10 },

  card: {
    backgroundColor: '#fff',
    borderRadius: 14,
    padding: 18,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#F0F0F0',
    gap: 8,
  },
  cardLabel: { fontSize: 12, color: '#999' },
  currentRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  bigPct: { fontSize: 32, fontWeight: '800' },
  changeRow: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  changeText: { fontSize: 13, color: '#444' },
  body: { fontSize: 14, lineHeight: 21, color: '#444' },

  badge: { paddingHorizontal: 10, paddingVertical: 3, borderRadius: 20 },
  badgeText: { fontSize: 12, fontWeight: '700' },

  chipRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  chip: { backgroundColor: '#F5F5F5', borderRadius: 12, paddingHorizontal: 8, paddingVertical: 3 },
  chipText: { fontSize: 12, fontWeight: '600' },

  empty: { fontSize: 13, color: '#888', lineHeight: 20 },
  axis: { flexDirection: 'row', justifyContent: 'space-between', marginLeft: 36 },
  chartWrap: { flexDirection: 'row', gap: 6 },
  yAxis: { width: 30, height: 120, marginTop: 6, justifyContent: 'space-between', alignItems: 'flex-end' },
  flex1: { flex: 1 },
  axisText: { fontSize: 12, color: '#888' },

  pointRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingVertical: 8,
    paddingHorizontal: 6,
    borderTopWidth: 1,
    borderTopColor: '#F2F2F2',
  },
  pointRowCurrent: { backgroundColor: '#F2FBF9', borderRadius: 8 },
  pointDate: { flex: 1, fontSize: 13, color: '#555' },
  pointPct: { fontSize: 14, fontWeight: '700', width: 56, textAlign: 'right' },

});
