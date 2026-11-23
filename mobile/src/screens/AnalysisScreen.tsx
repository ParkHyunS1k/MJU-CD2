import React, { useEffect, useRef, useState } from 'react';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import {
  Animated,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import { api } from '../lib/api';
import type { InBodyStackParamList } from '../navigation/types';

type Nav = NativeStackNavigationProp<InBodyStackParamList, 'AnalysisScreen'>;
type Rt = RouteProp<InBodyStackParamList, 'AnalysisScreen'>;

const TEAL = '#00C2A8';

export default function AnalysisScreen() {
  const nav = useNavigation<Nav>();
  const { id } = useRoute<Rt>().params;
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const bar1 = useRef(new Animated.Value(0)).current;
  const bar2 = useRef(new Animated.Value(0)).current;
  const bar3 = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(bar1, { toValue: 0.85, duration: 2200, useNativeDriver: false }),
      Animated.timing(bar2, { toValue: 0.7, duration: 3000, useNativeDriver: false }),
      Animated.timing(bar3, { toValue: 0.9, duration: 3800, useNativeDriver: false }),
    ]).start();

    api.analyzeInBody(id)
      .then(() => {
        Animated.parallel([
          Animated.timing(bar1, { toValue: 1, duration: 400, useNativeDriver: false }),
          Animated.timing(bar2, { toValue: 1, duration: 400, useNativeDriver: false }),
          Animated.timing(bar3, { toValue: 1, duration: 400, useNativeDriver: false }),
        ]).start(() => setDone(true));
      })
      .catch((e: unknown) => {
        setError(e instanceof Error ? e.message : '분석 실패');
        setDone(true);
      });
  }, [id]);

  const onConfirm = () => {
    if (error) {
      nav.goBack();
    } else {
      nav.navigate('ReportScreen', { id });
    }
  };

  const barWidth = (anim: Animated.Value) =>
    anim.interpolate({ inputRange: [0, 1], outputRange: ['0%', '100%'] });

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity style={styles.backBtn} onPress={() => nav.goBack()}>
          <Ionicons name="arrow-back" size={22} color="#fff" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>데이터 분석 중</Text>
      </View>

      <View style={styles.body}>
        <View style={styles.card}>
          <Text style={styles.cardTitle}>분석 진행 중</Text>
          <Text style={styles.cardSub}>업로드된 데이터를 분석하고 있습니다.</Text>

          <View style={styles.bars}>
            {[bar1, bar2, bar3].map((bar, i) => (
              <View key={i} style={styles.barBg}>
                <Animated.View style={[styles.barFill, { width: barWidth(bar) }]} />
              </View>
            ))}
          </View>

          {error ? (
            <Text style={styles.errorText}>{error}</Text>
          ) : (
            <Text style={styles.desc}>
              분석이 완료되면 위험도 평가 결과를 확인할 수 있습니다.{'\n'}이는 약 1분 정도 소요될 수 있습니다.
            </Text>
          )}

          <TouchableOpacity
            style={[styles.confirmBtn, done && styles.confirmBtnActive]}
            onPress={onConfirm}
            disabled={!done}
          >
            <Text style={[styles.confirmBtnText, done && styles.confirmBtnTextActive]}>
              분석 완료 확인
            </Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F5F5' },

  header: {
    backgroundColor: TEAL,
    flexDirection: 'row',
    alignItems: 'center',
    paddingTop: 56,
    paddingBottom: 20,
    paddingHorizontal: 16,
  },
  backBtn: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: 'rgba(0,0,0,0.15)',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  headerTitle: { color: '#fff', fontSize: 18, fontWeight: '700' },

  body: { flex: 1, padding: 20 },
  card: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 28,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 8,
    elevation: 3,
  },
  cardTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1A1A1A',
    marginBottom: 8,
    textAlign: 'center',
  },
  cardSub: {
    fontSize: 14,
    color: '#888',
    textAlign: 'center',
    marginBottom: 28,
  },

  bars: { gap: 10, marginBottom: 24 },
  barBg: {
    height: 8,
    backgroundColor: '#E5E5E5',
    borderRadius: 4,
    overflow: 'hidden',
  },
  barFill: {
    height: '100%',
    backgroundColor: TEAL,
    borderRadius: 4,
  },

  desc: {
    fontSize: 13,
    color: '#888',
    textAlign: 'center',
    lineHeight: 20,
    marginBottom: 28,
  },
  errorText: {
    fontSize: 13,
    color: '#EF5350',
    textAlign: 'center',
    marginBottom: 28,
  },

  confirmBtn: {
    borderWidth: 2,
    borderColor: '#CCCCCC',
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: 'center',
  },
  confirmBtnActive: { borderColor: TEAL },
  confirmBtnText: { fontSize: 16, fontWeight: '700', color: '#CCCCCC' },
  confirmBtnTextActive: { color: TEAL },
});
