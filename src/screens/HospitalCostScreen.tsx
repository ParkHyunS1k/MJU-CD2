import React from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

const TEAL = '#00C2A8';

const CRITERIA = [
  '위험도 기반 병원 방문 빈도 추정',
  '진료과별 평균 비용 데이터',
  '건강보험 적용 기준',
];

export default function HospitalCostScreen() {
  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.content}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>병원비 예측</Text>
        <Text style={styles.headerSub}>위험도 기반 의료비 예측</Text>
      </View>

      <View style={styles.card}>
        <Ionicons name="cash-outline" size={48} color={TEAL} style={styles.icon} />
        <Text style={styles.cardTitle}>병원비 예측 분석</Text>
        <Text style={styles.cardDesc}>
          인바디 분석 결과를 바탕으로{'\n'}향후 발생 가능한 의료비를 예측합니다.
        </Text>
        <Text style={styles.cardNote}>
          최신 인바디 분석 결과가 있으면{'\n'}예상 연료비를 확인할 수 있습니다.
        </Text>
      </View>

      <View style={styles.infoSection}>
        <Text style={styles.infoTitle}>예측 기준</Text>
        {CRITERIA.map((item) => (
          <View key={item} style={styles.infoRow}>
            <Ionicons name="checkmark-circle-outline" size={16} color={TEAL} />
            <Text style={styles.infoText}>{item}</Text>
          </View>
        ))}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: '#F5F5F5' },
  content: { paddingBottom: 32 },

  header: {
    backgroundColor: TEAL,
    paddingTop: 60,
    paddingBottom: 28,
    paddingHorizontal: 20,
    borderBottomLeftRadius: 24,
    borderBottomRightRadius: 24,
  },
  headerTitle: { color: '#fff', fontSize: 24, fontWeight: '700', marginBottom: 4 },
  headerSub: { color: 'rgba(255,255,255,0.85)', fontSize: 14 },

  card: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 32,
    margin: 16,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 8,
    elevation: 3,
    borderWidth: 1,
    borderColor: '#F0F0F0',
  },
  icon: { marginBottom: 16 },
  cardTitle: { fontSize: 18, fontWeight: '700', color: '#1A1A1A', marginBottom: 12, textAlign: 'center' },
  cardDesc: { fontSize: 14, color: '#555', textAlign: 'center', lineHeight: 22, marginBottom: 12 },
  cardNote: { fontSize: 13, color: '#999', textAlign: 'center', lineHeight: 20 },

  infoSection: { marginHorizontal: 16, marginTop: 8 },
  infoTitle: { fontSize: 16, fontWeight: '700', color: '#1A1A1A', marginBottom: 12 },
  infoRow: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 10 },
  infoText: { fontSize: 14, color: '#555' },
});
