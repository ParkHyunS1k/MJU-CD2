import React, { useCallback, useState } from 'react';
import { useFocusEffect } from '@react-navigation/native';
import {
  ActivityIndicator,
  Alert,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';

import { useAuth } from '../contexts/AuthContext';
import { api } from '../lib/api';
import type { InBodyRecord, UserProfile } from '../types/models';

export default function HomeScreen() {
  const { signOut } = useAuth();
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [latest, setLatest] = useState<InBodyRecord | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    try {
      const [me, logs] = await Promise.all([
        api.getMe().catch(() => null),
        api.listInBody(1).catch(() => []),
      ]);
      setProfile(me);
      setLatest(logs[0] ?? null);
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : '불러오기 실패';
      Alert.alert('오류', message);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      load();
    }, [load]),
  );

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" />
      </View>
    );
  }

  return (
    <ScrollView
      contentContainerStyle={styles.container}
      refreshControl={
        <RefreshControl
          refreshing={refreshing}
          onRefresh={() => {
            setRefreshing(true);
            load();
          }}
        />
      }
    >
      <View style={styles.card}>
        <Text style={styles.cardLabel}>내 정보</Text>
        <Text style={styles.cardTitle}>
          {profile?.nickname ?? '닉네임 없음'}
        </Text>
        <Text style={styles.cardSub}>
          {profile?.gender ?? '-'} · {profile?.birth_year ?? '-'}년생
        </Text>
        <Text style={styles.cardSub}>
          목표: {profile?.fitness_goal ?? '미설정'}
        </Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardLabel}>최근 InBody</Text>
        {latest ? (
          <>
            <Text style={styles.cardTitle}>{latest.weight} kg</Text>
            <Text style={styles.cardSub}>
              체지방률 {latest.body_fat_percentage ?? '-'}% · 골격근{' '}
              {latest.skeletal_muscle_mass ?? '-'}kg
            </Text>
            <Text style={styles.cardSub}>
              측정일{' '}
              {latest.measured_at
                ? new Date(latest.measured_at).toLocaleDateString()
                : '-'}
            </Text>
          </>
        ) : (
          <Text style={styles.cardSub}>아직 기록이 없습니다.</Text>
        )}
      </View>

      <TouchableOpacity
        style={styles.signOut}
        onPress={() => {
          Alert.alert('로그아웃', '정말 로그아웃 하시겠습니까?', [
            { text: '취소', style: 'cancel' },
            { text: '로그아웃', style: 'destructive', onPress: signOut },
          ]);
        }}
      >
        <Text style={styles.signOutText}>로그아웃</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { padding: 16, gap: 12 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  card: {
    backgroundColor: '#f6f8fa',
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e1e4e8',
  },
  cardLabel: { fontSize: 12, color: '#666', marginBottom: 4 },
  cardTitle: { fontSize: 22, fontWeight: '700', marginBottom: 4 },
  cardSub: { fontSize: 14, color: '#444', marginTop: 2 },
  signOut: {
    marginTop: 24,
    alignItems: 'center',
    padding: 12,
  },
  signOutText: { color: '#d73a49', fontWeight: '600' },
});
