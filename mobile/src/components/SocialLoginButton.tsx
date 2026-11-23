import React from 'react';
import { Image, StyleSheet, Text, TouchableOpacity, View } from 'react-native';

// 로고는 각 회사가 제공하는 공식 이미지를 assets/social/ 에 넣어 사용한다.
// (카카오: Kakao Developers 디자인 가이드 / 구글: Sign in with Google 브랜딩 가이드)
const LOGOS = {
  kakao: require('../../assets/social/kakao.png'),
  google: require('../../assets/social/google.png'),
};

type Provider = keyof typeof LOGOS;

interface Props {
  provider: Provider;
  label: string;
  onPress?: () => void; // 소셜 로그인 연결 전까지는 비워 둔다
  disabled?: boolean;
}

export default function SocialLoginButton({ provider, label, onPress, disabled }: Props) {
  const isKakao = provider === 'kakao';
  return (
    <TouchableOpacity
      style={[styles.button, isKakao ? styles.kakao : styles.google, disabled && styles.disabled]}
      onPress={onPress}
      disabled={disabled}
      accessibilityRole="button"
      accessibilityLabel={label}
    >
      <View style={styles.logoSlot}>
        <Image source={LOGOS[provider]} style={styles.logo} resizeMode="contain" />
      </View>
      <Text style={[styles.label, isKakao ? styles.kakaoLabel : styles.googleLabel]}>
        {label}
      </Text>
      {/* 로고 칸과 같은 너비의 빈 칸을 오른쪽에 둬서 글자가 버튼 정중앙에 오게 한다 */}
      <View style={styles.logoSlot} />
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: 12,
    paddingVertical: 13,
    paddingHorizontal: 16,
    marginBottom: 10,
  },
  // 카카오 가이드: 배경 #FEE500, 글자 검정 85%
  kakao: {
    backgroundColor: '#FEE500',
  },
  // 구글 가이드: 흰 배경, 회색 테두리
  google: {
    backgroundColor: '#FFFFFF',
    borderWidth: 1,
    borderColor: '#DADCE0',
  },
  disabled: { opacity: 0.5 },
  logoSlot: {
    width: 22,
    alignItems: 'center',
  },
  logo: {
    width: 20,
    height: 20,
  },
  label: {
    flex: 1,
    textAlign: 'center',
    fontSize: 15,
    fontWeight: '600',
  },
  kakaoLabel: { color: 'rgba(0,0,0,0.85)' },
  googleLabel: { color: '#1F1F1F' },
});
