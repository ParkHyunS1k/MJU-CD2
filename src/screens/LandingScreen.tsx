import React from 'react';
import {
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { AuthStackParamList } from '../navigation/types';

type NavProp = NativeStackNavigationProp<AuthStackParamList>;

const FEATURES = [
  {
    icon: 'document-text-outline' as const,
    title: '인바디 결과지 업로드',
    desc: '인바디 결과지를 사진이나 PDF로 올리면 AI가 수치를 읽고 분석합니다.',
  },
  {
    icon: 'trending-up-outline' as const,
    title: '위험도 리포트',
    desc: '주요 수치 위험 수준을 한눈에 확인하고 개선 방향을 안내받습니다.',
  },
];

export default function LandingScreen() {
  const navigation = useNavigation<NavProp>();

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.content}>
      <View style={styles.header}>
        <Text style={styles.headerTag}>AI HEALTH RISK ASSESSMENT</Text>
      </View>

      <View style={styles.hero}>
        <Text style={styles.heroTitle}>{'내 몸의 위험 신호를\n먼저 알아보세요'}</Text>
        <Text style={styles.heroDesc}>
          AI 기반 건강 위험도 분석으로 질병을 예방하고 건강을 지키세요.
        </Text>

        <TouchableOpacity
          style={styles.primaryButton}
          onPress={() => navigation.navigate('Register')}
        >
          <Text style={styles.primaryButtonText}>시작하기 — 무료로 가입</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.secondaryButton}
          onPress={() => navigation.navigate('Login')}
        >
          <Text style={styles.secondaryButtonText}>이미 계정이 있으신가요? 로그인</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.features}>
        <Text style={styles.featuresTitle}>주요 기능</Text>
        {FEATURES.map((f) => (
          <View key={f.title} style={styles.featureCard}>
            <View style={styles.featureIcon}>
              <Ionicons name={f.icon} size={22} color="#00C2A8" />
            </View>
            <View style={styles.featureText}>
              <Text style={styles.featureTitle}>{f.title}</Text>
              <Text style={styles.featureDesc}>{f.desc}</Text>
            </View>
          </View>
        ))}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: '#F5F5F5' },
  content: { paddingBottom: 40 },

  header: {
    backgroundColor: '#00C2A8',
    borderBottomLeftRadius: 24,
    borderBottomRightRadius: 24,
    paddingTop: 60,
    paddingBottom: 28,
    paddingHorizontal: 24,
    alignItems: 'center',
  },
  headerTag: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '700',
    letterSpacing: 1.5,
  },

  hero: {
    backgroundColor: '#fff',
    marginHorizontal: 16,
    marginTop: -1,
    borderRadius: 16,
    padding: 24,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 8,
    elevation: 3,
  },
  heroTitle: {
    fontSize: 24,
    fontWeight: '700',
    color: '#1A1A1A',
    lineHeight: 34,
    textAlign: 'center',
    marginBottom: 12,
  },
  heroDesc: {
    fontSize: 13,
    color: '#757575',
    textAlign: 'center',
    lineHeight: 20,
    marginBottom: 24,
  },

  primaryButton: {
    backgroundColor: '#00C2A8',
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
    marginBottom: 10,
  },
  primaryButtonText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '700',
  },
  secondaryButton: {
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#E5E5E5',
  },
  secondaryButtonText: {
    color: '#555',
    fontSize: 14,
  },

  features: {
    paddingHorizontal: 16,
    paddingTop: 8,
  },
  featuresTitle: {
    fontSize: 17,
    fontWeight: '700',
    color: '#1A1A1A',
    marginBottom: 12,
  },
  featureCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#F0F0F0',
  },
  featureIcon: {
    width: 40,
    height: 40,
    borderRadius: 10,
    backgroundColor: '#F0FAF8',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 14,
  },
  featureText: { flex: 1 },
  featureTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: '#1A1A1A',
    marginBottom: 4,
  },
  featureDesc: {
    fontSize: 13,
    color: '#757575',
    lineHeight: 18,
  },
});
