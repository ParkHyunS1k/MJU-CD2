import React, { useCallback, useState } from 'react';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import { ActivityIndicator, ScrollView, StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import ReportView from '../components/report/ReportView';
import { api } from '../lib/api';
import type { InBodyRecord } from '../types/models';
import { formatDate } from './ReportScreen';

const TEAL = '#00C2A8';
// 최근 기록 중 리포트가 있는 것을 찾을 범위
const SEARCH_LIMIT = 20;

// 하단 "리포트" 탭: 리포트가 있는 기록 중 가장 최근 것을 보여 준다.
// (가장 최근 기록 하나만 보면, 분석하지 않은 기록이 끼어 있을 때 리포트가 없다고 나오는 문제가 있었다)
export default function ReportTabScreen() {
  // 리포트 탭은 기록 탭과 다른 화면 묶음이라, 위험도 상세는 기록 탭(Records) 안에서 연다
  const navigation = useNavigation<any>();
  const [record, setRecord] = useState<InBodyRecord | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // 탭에 들어올 때마다 다시 불러와서, 방금 분석한 결과도 바로 보이게 한다
  useFocusEffect(
    useCallback(() => {
      let active = true;
      api
        .listInBody(SEARCH_LIMIT)
        .then((records) => {
          if (!active) return;
          setRecord(records.find((r) => r.ai_report) ?? null);
          setError(null);
        })
        .catch((e) => active && setError(e instanceof Error ? e.message : '불러오기 실패'))
        .finally(() => active && setLoading(false));
      return () => {
        active = false;
      };
    }, []),
  );

  const header = (
    <View style={styles.header}>
      <Text style={styles.headerTitle}>건강 리포트</Text>
    </View>
  );

  if (loading) {
    return (
      <View style={styles.page}>
        {header}
        <View style={styles.center}>
          <ActivityIndicator size="large" color={TEAL} />
        </View>
      </View>
    );
  }

  if (!record?.ai_report) {
    return (
      <View style={styles.page}>
        {header}
        <View style={styles.center}>
          <Ionicons name="document-text-outline" size={64} color="#CCCCCC" />
          <Text style={styles.emptyTitle}>{error ? '리포트를 불러오지 못했어요' : '리포트가 없습니다'}</Text>
          <Text style={styles.emptyDesc}>
            {error ?? '인바디 파일을 업로드하고\nAI 분석을 시작하세요.'}
          </Text>
        </View>
      </View>
    );
  }

  return (
    <ScrollView style={styles.page} contentContainerStyle={styles.content}>
      {header}
      <View style={styles.body}>
        <ReportView
          report={record.ai_report}
          dateLabel={formatDate(record.measured_at)}
          onPressDisease={(disease) =>
            navigation.navigate('Records', {
              screen: 'RiskScreen',
              params: { id: record.id, disease },
              initial: false,
            })
          }
        />
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  page: { flex: 1, backgroundColor: '#F5F5F5' },
  content: { paddingBottom: 40 },
  body: { padding: 16 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 14, padding: 40 },
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
  headerTitle: { fontSize: 20, fontWeight: '700', color: '#fff' },
});
