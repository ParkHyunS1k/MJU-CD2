import React, { useCallback, useState } from 'react';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import {
  ActivityIndicator,
  Alert,
  FlatList,
  RefreshControl,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';

import { Ionicons } from '@expo/vector-icons';

import { api } from '../lib/api';
import type { InBodyStackParamList } from '../navigation/types';
import type { InBodyRecord } from '../types/models';

type Nav = NativeStackNavigationProp<InBodyStackParamList, 'InBodyList'>;

export default function InBodyListScreen() {
  const nav = useNavigation<Nav>();
  const [items, setItems] = useState<InBodyRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    try {
      const data = await api.listInBody(50);
      setItems(data);
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
    <View style={styles.container}>
      <FlatList
        data={items}
        keyExtractor={(item) => item.id}
        contentContainerStyle={items.length === 0 ? styles.emptyWrap : { padding: 12 }}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={() => {
              setRefreshing(true);
              load();
            }}
          />
        }
        ListEmptyComponent={
          <Text style={styles.empty}>{'기록이 없습니다.\n오른쪽 아래 카메라 버튼으로 결과지를 올리거나\n+ 버튼으로 직접 입력하세요.'}</Text>
        }
        renderItem={({ item }) => (
          <TouchableOpacity
            style={styles.row}
            onPress={() => nav.navigate('InBodyDetail', { id: item.id })}
          >
            <Text style={styles.rowTitle}>{item.weight} kg</Text>
            <Text style={styles.rowSub}>
              체지방률 {item.body_fat_percentage ?? '-'}% · 골격근{' '}
              {item.skeletal_muscle_mass ?? '-'}kg
            </Text>
            <Text style={styles.rowDate}>
              {item.measured_at
                ? new Date(item.measured_at).toLocaleString()
                : '-'}
            </Text>
          </TouchableOpacity>
        )}
      />

      <TouchableOpacity
        style={[styles.fab, styles.fabSecondary]}
        onPress={() => nav.navigate('InBodyCreate')}
      >
        <Ionicons name="add" size={28} color="#00A38D" />
      </TouchableOpacity>

      <TouchableOpacity
        style={styles.fab}
        onPress={() => nav.navigate('InBodyUpload')}
      >
        <Ionicons name="camera" size={26} color="#fff" />
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  emptyWrap: { flexGrow: 1, alignItems: 'center', justifyContent: 'center', padding: 32 },
  empty: { color: '#666', textAlign: 'center', lineHeight: 22 },
  row: {
    padding: 14,
    backgroundColor: '#f6f8fa',
    borderRadius: 10,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#e1e4e8',
  },
  rowTitle: { fontSize: 18, fontWeight: '700' },
  rowSub: { fontSize: 13, color: '#444', marginTop: 4 },
  rowDate: { fontSize: 12, color: '#888', marginTop: 4 },
  fab: {
    position: 'absolute',
    right: 20,
    bottom: 24,
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: '#00C2A8',
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
    elevation: 4,
  },
  fabSecondary: {
    bottom: 92,
    backgroundColor: '#fff',
    borderWidth: 1.5,
    borderColor: '#00C2A8',
  },
});
