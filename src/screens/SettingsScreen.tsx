import React from 'react';
import { Alert, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';

import { useAuth } from '../contexts/AuthContext';
import type { SettingsStackParamList } from '../navigation/types';

const TEAL = '#00C2A8';

type Nav = NativeStackNavigationProp<SettingsStackParamList, 'SettingsMain'>;

// go가 없는 메뉴는 아직 화면이 없다
type Item = {
  icon: keyof typeof Ionicons.glyphMap;
  label: string;
  sub: string;
  go?: (navigation: Nav) => void;
};

const ITEMS: Item[] = [
  {
    icon: 'person-outline',
    label: '프로필 관리',
    sub: '이름, 나이, 목표 설정',
    go: (n) => n.navigate('ProfileEdit'),
  },
  {
    icon: 'shield-checkmark-outline',
    label: '개인정보 처리방침',
    sub: '',
    go: (n) => n.navigate('Policy', { kind: 'privacy' }),
  },
  {
    icon: 'document-text-outline',
    label: '이용약관',
    sub: '',
    go: (n) => n.navigate('Policy', { kind: 'terms' }),
  },
];

export default function SettingsScreen() {
  const { signOut } = useAuth();
  const navigation = useNavigation<Nav>();

  const onSignOut = () => {
    Alert.alert('로그아웃', '정말 로그아웃 하시겠습니까?', [
      { text: '취소', style: 'cancel' },
      { text: '로그아웃', style: 'destructive', onPress: signOut },
    ]);
  };

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.content}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>설정</Text>
      </View>

      <View style={styles.section}>
        {ITEMS.map((item, idx) => (
          <TouchableOpacity
            key={item.label}
            style={[styles.row, idx < ITEMS.length - 1 && styles.rowBorder]}
            onPress={item.go ? () => item.go!(navigation) : undefined}
          >
            <View style={styles.rowLeft}>
              <View style={styles.rowIconWrap}>
                <Ionicons name={item.icon} size={20} color="#555" />
              </View>
              <View>
                <Text style={styles.rowLabel}>{item.label}</Text>
                {item.sub ? <Text style={styles.rowSub}>{item.sub}</Text> : null}
              </View>
            </View>
            <Ionicons name="chevron-forward" size={18} color="#CCCCCC" />
          </TouchableOpacity>
        ))}
      </View>

      <TouchableOpacity style={styles.signOutBtn} onPress={onSignOut}>
        <Text style={styles.signOutText}>로그아웃</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: '#F5F5F5' },
  content: { paddingBottom: 40 },

  header: {
    backgroundColor: TEAL,
    paddingTop: 60,
    paddingBottom: 24,
    paddingHorizontal: 20,
    borderBottomLeftRadius: 24,
    borderBottomRightRadius: 24,
  },
  headerTitle: { color: '#fff', fontSize: 24, fontWeight: '700' },

  section: {
    backgroundColor: '#fff',
    borderRadius: 14,
    marginHorizontal: 16,
    marginTop: 20,
    borderWidth: 1,
    borderColor: '#F0F0F0',
    overflow: 'hidden',
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 16,
  },
  rowBorder: {
    borderBottomWidth: 1,
    borderBottomColor: '#F5F5F5',
  },
  rowLeft: { flexDirection: 'row', alignItems: 'center' },
  rowIconWrap: { marginRight: 12 },
  rowLabel: { fontSize: 15, color: '#1A1A1A' },
  rowSub: { fontSize: 12, color: '#AAAAAA', marginTop: 2 },

  signOutBtn: {
    marginHorizontal: 16,
    marginTop: 16,
    backgroundColor: '#fff',
    borderRadius: 14,
    padding: 16,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#F0F0F0',
  },
  signOutText: { fontSize: 15, color: '#FF3B30', fontWeight: '600' },
});
