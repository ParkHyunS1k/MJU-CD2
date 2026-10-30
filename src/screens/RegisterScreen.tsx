import React, { useState } from 'react';
import {
  Alert,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';

import SocialLoginButton from '../components/SocialLoginButton';
import { useAuth } from '../contexts/AuthContext';
import type { AuthStackParamList } from '../navigation/types';

type NavProp = NativeStackNavigationProp<AuthStackParamList>;

const TEAL = '#00C2A8';

// 화면에 안내한 비밀번호 조건. 안내 문구와 실제 검사가 같은 목록을 쓴다.
const PASSWORD_RULES: { label: string; test: (pw: string) => boolean }[] = [
  { label: '8자 이상', test: (pw) => pw.length >= 8 },
  { label: '영문 포함', test: (pw) => /[A-Za-z]/.test(pw) },
  { label: '숫자 포함', test: (pw) => /\d/.test(pw) },
];

export default function RegisterScreen() {
  const { signUp } = useAuth();
  const navigation = useNavigation<NavProp>();

  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPw, setConfirmPw] = useState('');
  const [showPw, setShowPw] = useState(false);
  const [showConfirmPw, setShowConfirmPw] = useState(false);
  const [agreed, setAgreed] = useState(false);
  const [busy, setBusy] = useState(false);

  const passwordValid = PASSWORD_RULES.every((r) => r.test(password));
  const passwordsMatch = confirmPw.length > 0 && password === confirmPw;
  const canSubmit =
    agreed && !busy && !!name.trim() && !!email.trim() && passwordValid && passwordsMatch;

  const onRegister = async () => {
    if (!name || !email || !password || !confirmPw) {
      Alert.alert('모든 항목을 입력해주세요.');
      return;
    }
    const failed = PASSWORD_RULES.filter((r) => !r.test(password)).map((r) => r.label);
    if (failed.length > 0) {
      Alert.alert('비밀번호 조건을 확인해 주세요', `${failed.join(', ')} 조건이 필요해요.`);
      return;
    }
    if (password !== confirmPw) {
      Alert.alert('비밀번호가 일치하지 않습니다.');
      return;
    }
    if (!agreed) {
      Alert.alert('이용약관에 동의해주세요.');
      return;
    }
    setBusy(true);
    try {
      await signUp(email.trim(), password);
      Alert.alert('가입 완료', '이메일 확인이 필요할 수 있습니다.');
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : '가입에 실패했습니다.';
      Alert.alert('실패', message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={styles.flex}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <ScrollView style={styles.flex} contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
        <View style={styles.header}>
          <Text style={styles.headerTitle}>회원가입</Text>
          <Text style={styles.headerSubtitle}>건강 관리 여정을 시작하세요</Text>
        </View>

        <View style={styles.form}>
          <Text style={styles.label}>이름</Text>
          <TextInput
            style={styles.input}
            placeholder="이름을 입력하세요"
            placeholderTextColor="#BBBBBB"
            value={name}
            onChangeText={setName}
            editable={!busy}
          />

          <Text style={styles.label}>이메일</Text>
          <TextInput
            style={styles.input}
            placeholder="이메일을 입력하세요"
            placeholderTextColor="#BBBBBB"
            autoCapitalize="none"
            keyboardType="email-address"
            value={email}
            onChangeText={setEmail}
            editable={!busy}
          />

          <Text style={styles.label}>비밀번호</Text>
          <View style={styles.inputRow}>
            <TextInput
              style={styles.inputFlex}
              placeholder="비밀번호를 입력하세요"
              placeholderTextColor="#BBBBBB"
              secureTextEntry={!showPw}
              value={password}
              onChangeText={setPassword}
              editable={!busy}
            />
            <TouchableOpacity onPress={() => setShowPw(!showPw)} style={styles.eyeBtn}>
              <Ionicons name={showPw ? 'eye-outline' : 'eye-off-outline'} size={20} color="#999" />
            </TouchableOpacity>
          </View>
          <View style={styles.ruleRow}>
            {PASSWORD_RULES.map((rule) => {
              const ok = rule.test(password);
              return (
                <View key={rule.label} style={styles.ruleItem}>
                  <Ionicons
                    name={ok ? 'checkmark-circle' : 'ellipse-outline'}
                    size={14}
                    color={ok ? TEAL : '#BBBBBB'}
                  />
                  <Text style={[styles.ruleText, ok && styles.ruleTextOk]}>{rule.label}</Text>
                </View>
              );
            })}
          </View>

          <Text style={styles.label}>비밀번호 확인</Text>
          <View style={styles.inputRow}>
            <TextInput
              style={styles.inputFlex}
              placeholder="비밀번호를 다시 입력하세요"
              placeholderTextColor="#BBBBBB"
              secureTextEntry={!showConfirmPw}
              value={confirmPw}
              onChangeText={setConfirmPw}
              editable={!busy}
            />
            <TouchableOpacity onPress={() => setShowConfirmPw(!showConfirmPw)} style={styles.eyeBtn}>
              <Ionicons name={showConfirmPw ? 'eye-outline' : 'eye-off-outline'} size={20} color="#999" />
            </TouchableOpacity>
          </View>
          {confirmPw.length > 0 && (
            <Text style={[styles.matchText, passwordsMatch ? styles.matchOk : styles.matchBad]}>
              {passwordsMatch ? '비밀번호가 일치해요' : '비밀번호가 일치하지 않아요'}
            </Text>
          )}

          <TouchableOpacity
            style={styles.termsRow}
            onPress={() => setAgreed(!agreed)}
          >
            <View style={[styles.checkbox, agreed && styles.checkboxChecked]}>
              {agreed && <Ionicons name="checkmark" size={12} color="#fff" />}
            </View>
            <Text style={styles.termsText}>이용약관 및 개인정보처리방침에 동의합니다</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.primaryButton, canSubmit ? styles.primaryButtonActive : styles.primaryButtonInactive]}
            onPress={onRegister}
            disabled={!canSubmit}
          >
            <Text style={styles.primaryButtonText}>가입하기</Text>
          </TouchableOpacity>

          <View style={styles.divider}>
            <View style={styles.dividerLine} />
            <Text style={styles.dividerText}>또는</Text>
            <View style={styles.dividerLine} />
          </View>

          <SocialLoginButton
            provider="kakao"
            label="카카오 로그인"
          />
          <SocialLoginButton
            provider="google"
            label="Google 로그인"
          />

          <View style={styles.loginRow}>
            <Text style={styles.loginText}>이미 계정이 있으신가요? </Text>
            <TouchableOpacity onPress={() => navigation.navigate('Login')}>
              <Text style={styles.loginLink}>로그인</Text>
            </TouchableOpacity>
          </View>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: '#F5F5F5' },
  content: { flexGrow: 1 },

  header: {
    backgroundColor: '#00C2A8',
    borderBottomLeftRadius: 24,
    borderBottomRightRadius: 24,
    paddingTop: 70,
    paddingBottom: 32,
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 26,
    fontWeight: '700',
    color: '#fff',
    marginBottom: 6,
  },
  headerSubtitle: {
    fontSize: 14,
    color: 'rgba(255,255,255,0.85)',
  },

  form: {
    backgroundColor: '#fff',
    marginHorizontal: 16,
    marginTop: 20,
    borderRadius: 16,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 8,
    elevation: 3,
    marginBottom: 32,
  },

  label: {
    fontSize: 13,
    fontWeight: '600',
    color: '#333',
    marginBottom: 6,
    marginTop: 4,
  },
  input: {
    borderWidth: 1,
    borderColor: '#E5E5E5',
    borderRadius: 10,
    paddingHorizontal: 14,
    paddingVertical: 12,
    fontSize: 14,
    color: '#1A1A1A',
    backgroundColor: '#FAFAFA',
    marginBottom: 14,
  },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#E5E5E5',
    borderRadius: 10,
    backgroundColor: '#FAFAFA',
    marginBottom: 6,
  },
  inputFlex: {
    flex: 1,
    paddingHorizontal: 14,
    paddingVertical: 12,
    fontSize: 14,
    color: '#1A1A1A',
  },
  eyeBtn: {
    paddingHorizontal: 12,
    paddingVertical: 12,
  },
  ruleRow: { flexDirection: 'row', gap: 12, marginTop: 6, marginBottom: 14 },
  ruleItem: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  ruleText: { fontSize: 12, color: '#AAAAAA' },
  ruleTextOk: { color: TEAL, fontWeight: '600' },
  matchText: { fontSize: 12, marginTop: 6 },
  matchOk: { color: TEAL },
  matchBad: { color: '#E53935' },

  termsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#E5E5E5',
    borderRadius: 10,
    padding: 14,
    marginBottom: 16,
  },
  checkbox: {
    width: 18,
    height: 18,
    borderRadius: 4,
    borderWidth: 1.5,
    borderColor: '#CCCCCC',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 10,
  },
  checkboxChecked: {
    backgroundColor: '#00C2A8',
    borderColor: '#00C2A8',
  },
  termsText: {
    fontSize: 13,
    color: '#555',
    flex: 1,
  },

  primaryButton: {
    borderRadius: 12,
    paddingVertical: 15,
    alignItems: 'center',
    marginBottom: 16,
  },
  primaryButtonActive: { backgroundColor: '#00C2A8' },
  primaryButtonInactive: { backgroundColor: '#9E9E9E' },
  primaryButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },

  divider: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 14,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: '#E5E5E5',
  },
  dividerText: {
    marginHorizontal: 12,
    fontSize: 13,
    color: '#AAAAAA',
  },

  loginRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 8,
  },
  loginText: { fontSize: 13, color: '#888' },
  loginLink: { fontSize: 13, color: '#00C2A8', fontWeight: '600' },
});
