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

import { useAuth } from '../contexts/AuthContext';
import { friendlyAuthError } from '../lib/errors';
import type { AuthStackParamList } from '../navigation/types';

type NavProp = NativeStackNavigationProp<AuthStackParamList>;

export default function LoginScreen() {
  const { signIn } = useAuth();
  const navigation = useNavigation<NavProp>();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPw, setShowPw] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const [busy, setBusy] = useState(false);

  const onLogin = async () => {
    if (!email || !password) {
      Alert.alert('이메일과 비밀번호를 입력해주세요.');
      return;
    }
    setBusy(true);
    try {
      await signIn(email.trim(), password);
    } catch (e: unknown) {
      Alert.alert('로그인 실패', friendlyAuthError(e, '로그인에 실패했어요. 다시 시도해 주세요.'));
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
          <Text style={styles.headerTitle}>로그인</Text>
          <Text style={styles.headerSubtitle}>건강 관리를 시작하세요</Text>
        </View>

        <View style={styles.form}>
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

          <View style={styles.rememberRow}>
            <TouchableOpacity
              style={styles.checkRow}
              onPress={() => setRememberMe(!rememberMe)}
            >
              <View style={[styles.checkbox, rememberMe && styles.checkboxChecked]}>
                {rememberMe && <Ionicons name="checkmark" size={12} color="#fff" />}
              </View>
              <Text style={styles.rememberText}>로그인 유지</Text>
            </TouchableOpacity>
            <TouchableOpacity>
              <Text style={styles.forgotText}>비밀번호 찾기</Text>
            </TouchableOpacity>
          </View>

          <TouchableOpacity
            style={[styles.primaryButton, busy && styles.buttonDisabled]}
            onPress={onLogin}
            disabled={busy}
          >
            <Text style={styles.primaryButtonText}>로그인</Text>
          </TouchableOpacity>

          <View style={styles.signupRow}>
            <Text style={styles.signupText}>계정이 없으신가요? </Text>
            <TouchableOpacity onPress={() => navigation.navigate('Register')}>
              <Text style={styles.signupLink}>회원가입</Text>
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
    marginBottom: 14,
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

  rememberRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  checkRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  checkbox: {
    width: 18,
    height: 18,
    borderRadius: 4,
    borderWidth: 1.5,
    borderColor: '#CCCCCC',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 7,
  },
  checkboxChecked: {
    backgroundColor: '#00C2A8',
    borderColor: '#00C2A8',
  },
  rememberText: {
    fontSize: 13,
    color: '#555',
  },
  forgotText: {
    fontSize: 13,
    color: '#00C2A8',
  },

  primaryButton: {
    backgroundColor: '#00C2A8',
    borderRadius: 12,
    paddingVertical: 15,
    alignItems: 'center',
    marginBottom: 16,
  },
  buttonDisabled: { opacity: 0.6 },
  primaryButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },


  signupRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 8,
  },
  signupText: { fontSize: 13, color: '#888' },
  signupLink: { fontSize: 13, color: '#00C2A8', fontWeight: '600' },
});
