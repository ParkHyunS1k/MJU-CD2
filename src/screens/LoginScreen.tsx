import React, { useState } from 'react';
import { Alert, KeyboardAvoidingView, Platform, StyleSheet, View } from 'react-native';
import { Button, Text, TextInput, Surface } from 'react-native-paper';

import { useAuth } from '../contexts/AuthContext';

export default function LoginScreen() {
  const { signIn, signUp } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [mode, setMode] = useState<'signin' | 'signup'>('signin');
  const [busy, setBusy] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const onSubmit = async () => {
    if (!email || !password) {
      Alert.alert('이메일과 비밀번호를 입력해주세요.');
      return;
    }
    setBusy(true);
    try {
      if (mode === 'signin') {
        await signIn(email.trim(), password);
      } else {
        await signUp(email.trim(), password);
        Alert.alert('가입 완료', '이메일 확인이 필요할 수 있습니다.');
      }
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : '오류가 발생했습니다.';
      Alert.alert('실패', message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <View style={styles.header}>
        <Text variant="displaySmall" style={styles.title}>Healthcare</Text>
        <Text variant="bodyLarge" style={styles.subtitle}>
          {mode === 'signin' ? '계정에 로그인하세요' : '새 계정을 만드세요'}
        </Text>
      </View>

      <Surface style={styles.card} elevation={2}>
        <TextInput
          label="이메일"
          mode="outlined"
          autoCapitalize="none"
          keyboardType="email-address"
          value={email}
          onChangeText={setEmail}
          disabled={busy}
          left={<TextInput.Icon icon="email-outline" />}
          style={styles.input}
        />
        <TextInput
          label="비밀번호"
          mode="outlined"
          secureTextEntry={!showPassword}
          value={password}
          onChangeText={setPassword}
          disabled={busy}
          left={<TextInput.Icon icon="lock-outline" />}
          right={
            <TextInput.Icon
              icon={showPassword ? 'eye-off' : 'eye'}
              onPress={() => setShowPassword(v => !v)}
            />
          }
          style={styles.input}
        />

        <Button
          mode="contained"
          onPress={onSubmit}
          loading={busy}
          disabled={busy}
          style={styles.button}
          contentStyle={styles.buttonContent}
        >
          {mode === 'signin' ? '로그인' : '가입하기'}
        </Button>

        <Button
          mode="text"
          onPress={() => setMode(mode === 'signin' ? 'signup' : 'signin')}
          disabled={busy}
          style={styles.toggleButton}
        >
          {mode === 'signin'
            ? '계정이 없으신가요? 회원가입'
            : '이미 계정이 있으신가요? 로그인'}
        </Button>
      </Surface>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    padding: 24,
    backgroundColor: '#f5f5f5',
  },
  header: {
    alignItems: 'center',
    marginBottom: 32,
  },
  title: {
    fontWeight: '700',
    marginBottom: 6,
  },
  subtitle: {
    color: '#666',
  },
  card: {
    borderRadius: 16,
    padding: 24,
  },
  input: {
    marginBottom: 12,
  },
  button: {
    marginTop: 8,
    borderRadius: 8,
  },
  buttonContent: {
    paddingVertical: 6,
  },
  toggleButton: {
    marginTop: 8,
  },
});
