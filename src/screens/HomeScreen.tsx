import React, { useCallback, useState } from 'react';
import { useFocusEffect } from '@react-navigation/native';
import {
  ActivityIndicator,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';

import { api } from '../lib/api';
import type { InBodyRecord } from '../types/models';

const TEAL = '#00C2A8';

function getOverallRisk(record: InBodyRecord): string {
  const risks = record.ai_report?.risk_report;
  if (!risks?.length) return '분석 전'; // 리포트가 없으면 위험도를 알 수 없다
  if (risks.some((r) => r.risk_level === 'high')) return '높음';
  if (risks.some((r) => r.risk_level === 'medium')) return '주의';
  return '낮음';
}

function formatYearMonth(dateStr: string | null): string {
  if (!dateStr) return '날짜 없음';
  const d = new Date(dateStr);
  return `${d.getFullYear()}년 ${d.getMonth() + 1}월`;
}

export default function HomeScreen() {
  const navigation = useNavigation<any>();
  const [records, setRecords] = useState<InBodyRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    try {
      const data = await api.listInBody(5).catch(() => []);
      setRecords(data);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useFocusEffect(useCallback(() => { load(); }, [load]));

  const latest = records[0] ?? null;

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
      refreshControl={
        <RefreshControl
          refreshing={refreshing}
          onRefresh={() => { setRefreshing(true); load(); }}
          tintColor={TEAL}
        />
      }
    >
      <View style={styles.header}>
        <Text style={styles.headerGreet}>안녕하세요 👋</Text>
        <Text style={styles.headerTitle}>오늘의 건강 체크</Text>
      </View>

      {/* 내 건강 현황 */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>내 건강 현황</Text>
        <View style={styles.statusCard}>
          {latest ? (
            <View style={styles.statusRow}>
              <View style={styles.statusLeft}>
                <Text style={styles.statusLabel}>최근 분석</Text>
                <Text style={styles.statusMain}>
                  {formatYearMonth(latest.measured_at)} · 위험도 {getOverallRisk(latest)}
                </Text>
                <View style={styles.tagRow}>
                  {latest.ai_report?.risk_report?.some((r) => r.risk_level === 'medium') && (
                    <View style={[styles.tag, styles.tagOrange]}>
                      <Text style={[styles.tagText, styles.tagTextOrange]}>주의할 질환 있음</Text>
                    </View>
                  )}
                  {(latest.skeletal_muscle_mass ?? 0) >= 28 && (
                    <View style={[styles.tag, styles.tagGreen]}>
                      <Text style={[styles.tagText, styles.tagTextGreen]}>근육량 양호</Text>
                    </View>
                  )}
                </View>
              </View>
              <TouchableOpacity
                style={styles.resultBtn}
                onPress={() => navigation.navigate('Records', {
                  screen: 'InBodyDetail',
                  params: { id: latest.id },
                  initial: false, // 기록 목록을 아래에 깔아 두어 뒤로 가기가 목록으로 가게 한다
                })}
              >
                <Text style={styles.resultBtnText}>분석 결과{'\n'}보기</Text>
              </TouchableOpacity>
            </View>
          ) : (
            <Text style={styles.emptyText}>아직 분석 기록이 없습니다.</Text>
          )}
        </View>
      </View>

      {/* 인바디 파일 업로드 */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>인바디 파일 업로드</Text>
        <Text style={styles.uploadDesc}>
          최신 인바디 측정 결과를 업로드하여 건강 위험도를 분석해 드립니다.
        </Text>
        <TouchableOpacity
          style={styles.uploadBtn}
          onPress={() => navigation.navigate('Records', { screen: 'InBodyUpload', initial: false })}
        >
          <Text style={styles.uploadBtnText}>파일 업로드 시작</Text>
        </TouchableOpacity>
      </View>

      {/* 최근 기록 */}
      {records.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>최근 기록</Text>
          {records.slice(0, 3).map((rec) => (
            <TouchableOpacity
              key={rec.id}
              style={styles.recordRow}
              onPress={() => navigation.navigate('Records', {
                screen: 'InBodyDetail',
                params: { id: rec.id },
                initial: false,
              })}
            >
              <View>
                <Text style={styles.recordTitle}>{formatYearMonth(rec.measured_at)} 측정</Text>
                <Text style={styles.recordSub}>위험도: {getOverallRisk(rec)}</Text>
              </View>
              <Ionicons name="chevron-forward" size={18} color="#CCCCCC" />
            </TouchableOpacity>
          ))}
        </View>
      )}

    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: '#F5F5F5' },
  content: { paddingBottom: 32 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },

  header: {
    backgroundColor: TEAL,
    borderBottomLeftRadius: 20,
    borderBottomRightRadius: 20,
    paddingTop: 60,
    paddingBottom: 24,
    paddingHorizontal: 20,
  },
  headerGreet: { color: 'rgba(255,255,255,0.85)', fontSize: 14, marginBottom: 4 },
  headerTitle: { color: '#fff', fontSize: 22, fontWeight: '700' },

  section: { marginHorizontal: 16, marginTop: 20 },
  sectionTitle: { fontSize: 17, fontWeight: '700', color: '#1A1A1A', marginBottom: 12 },

  statusCard: {
    backgroundColor: '#fff',
    borderRadius: 14,
    padding: 16,
    borderWidth: 1,
    borderColor: '#F0F0F0',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  statusRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start' },
  statusLeft: { flex: 1 },
  statusLabel: { fontSize: 12, color: '#999', marginBottom: 4 },
  statusMain: { fontSize: 16, fontWeight: '700', color: '#1A1A1A', marginBottom: 10 },
  tagRow: { flexDirection: 'row', gap: 6, flexWrap: 'wrap' },
  tag: { backgroundColor: '#F0F0F0', paddingHorizontal: 8, paddingVertical: 3, borderRadius: 12 },
  tagText: { fontSize: 11, color: '#555' },
  tagOrange: { backgroundColor: '#FFF3E0' },
  tagTextOrange: { color: '#E65100' },
  tagGreen: { backgroundColor: '#E8F5E9' },
  tagTextGreen: { color: '#2E7D32' },
  resultBtn: {
    borderWidth: 1,
    borderColor: '#E5E5E5',
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 8,
    alignItems: 'center',
    marginLeft: 12,
  },
  resultBtnText: { fontSize: 12, color: '#333', textAlign: 'center', lineHeight: 17 },
  emptyText: { fontSize: 14, color: '#999' },

  uploadDesc: { fontSize: 13, color: '#757575', lineHeight: 20, marginBottom: 14 },
  uploadBtn: {
    backgroundColor: TEAL,
    borderRadius: 12,
    paddingVertical: 15,
    alignItems: 'center',
  },
  uploadBtnText: { color: '#fff', fontSize: 15, fontWeight: '700' },

  recordRow: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 8,
    borderWidth: 1,
    borderColor: '#F0F0F0',
  },
  recordTitle: { fontSize: 15, fontWeight: '600', color: '#1A1A1A', marginBottom: 3 },
  recordSub: { fontSize: 13, color: '#999' },

});

