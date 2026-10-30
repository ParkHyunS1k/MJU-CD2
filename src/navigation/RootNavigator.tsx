import React, { useEffect, useState } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';

import { useAuth } from '../contexts/AuthContext';
import SplashScreen from '../screens/SplashScreen';
import LandingScreen from '../screens/LandingScreen';
import LoginScreen from '../screens/LoginScreen';
import RegisterScreen from '../screens/RegisterScreen';
import HomeScreen from '../screens/HomeScreen';
import InBodyListScreen from '../screens/InBodyListScreen';
import InBodyCreateScreen from '../screens/InBodyCreateScreen';
import InBodyDetailScreen from '../screens/InBodyDetailScreen';
import InBodyUploadScreen from '../screens/InBodyUploadScreen';
import InBodyReviewScreen from '../screens/InBodyReviewScreen';
import AnalysisScreen from '../screens/AnalysisScreen';
import RiskScreen from '../screens/RiskScreen';
import ReportScreen from '../screens/ReportScreen';
import ReportTabScreen from '../screens/ReportTabScreen';
import SettingsScreen from '../screens/SettingsScreen';
import ProfileEditScreen from '../screens/ProfileEditScreen';
import PolicyScreen from '../screens/PolicyScreen';
import HealthSurveyScreen from '../screens/HealthSurveyScreen';
import { api } from '../lib/api';
import type { UserProfile } from '../types/models';
import type { AuthStackParamList, InBodyStackParamList, SettingsStackParamList } from './types';

const TEAL = '#00C2A8';

const AuthStack = createNativeStackNavigator<AuthStackParamList>();
const Tab = createBottomTabNavigator();
const InBodyStack = createNativeStackNavigator<InBodyStackParamList>();
const SettingsStack = createNativeStackNavigator<SettingsStackParamList>();

function AuthStackNavigator() {
  return (
    <AuthStack.Navigator screenOptions={{ headerShown: false }}>
      <AuthStack.Screen name="Landing" component={LandingScreen} />
      <AuthStack.Screen name="Login" component={LoginScreen} />
      <AuthStack.Screen name="Register" component={RegisterScreen} />
    </AuthStack.Navigator>
  );
}

function InBodyStackNavigator() {
  return (
    <InBodyStack.Navigator
      screenOptions={{
        headerTintColor: TEAL,
        headerTitleStyle: { fontWeight: '700' },
      }}
    >
      <InBodyStack.Screen
        name="InBodyList"
        component={InBodyListScreen}
        // 다른 탭(홈·리포트·설정)처럼 청록 헤더로 맞춘다
        options={{
          title: '기록',
          headerStyle: { backgroundColor: TEAL },
          headerTintColor: '#fff',
          headerShadowVisible: false,
        }}
      />
      <InBodyStack.Screen
        name="InBodyDetail"
        component={InBodyDetailScreen}
        options={{ title: '상세' }}
      />
      <InBodyStack.Screen
        name="InBodyCreate"
        component={InBodyCreateScreen}
        options={{ title: '직접 입력', presentation: 'modal' }}
      />
      <InBodyStack.Screen
        name="InBodyUpload"
        component={InBodyUploadScreen}
        options={{ headerShown: false }}
      />
      <InBodyStack.Screen
        name="InBodyReview"
        component={InBodyReviewScreen}
        options={{ title: '추출 결과 확인' }}
      />
      <InBodyStack.Screen
        name="AnalysisScreen"
        component={AnalysisScreen}
        options={{ headerShown: false }}
      />
      <InBodyStack.Screen
        name="RiskScreen"
        component={RiskScreen}
        options={{ title: '위험도 분석' }}
      />
      <InBodyStack.Screen
        name="ReportScreen"
        component={ReportScreen}
        options={{ title: '건강 리포트' }}
      />
    </InBodyStack.Navigator>
  );
}

function SettingsStackNavigator() {
  return (
    <SettingsStack.Navigator
      screenOptions={{
        headerTintColor: TEAL,
        headerTitleStyle: { fontWeight: '700' },
      }}
    >
      <SettingsStack.Screen
        name="SettingsMain"
        component={SettingsScreen}
        options={{ headerShown: false }}
      />
      <SettingsStack.Screen
        name="ProfileEdit"
        component={ProfileEditScreen}
        options={{ title: '프로필 관리' }}
      />
      <SettingsStack.Screen
        name="Policy"
        component={PolicyScreen}
        options={({ route }) => ({
          title: route.params.kind === 'privacy' ? '개인정보 처리방침' : '이용약관',
        })}
      />
    </SettingsStack.Navigator>
  );
}

type TabIconName = keyof typeof Ionicons.glyphMap;

const TAB_ICONS: Record<string, { active: TabIconName; inactive: TabIconName }> = {
  Home:     { active: 'home',          inactive: 'home-outline' },
  Report:   { active: 'document-text', inactive: 'document-text-outline' },
  Records:  { active: 'time',          inactive: 'time-outline' },
  Settings: { active: 'settings',      inactive: 'settings-outline' },
};

function MainTabs() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarActiveTintColor: TEAL,
        tabBarInactiveTintColor: '#AAAAAA',
        tabBarStyle: {
          borderTopColor: '#F0F0F0',
          backgroundColor: '#fff',
        },
        tabBarIcon: ({ focused, color, size }) => {
          const icons = TAB_ICONS[route.name];
          const name = focused ? icons?.active : icons?.inactive;
          return <Ionicons name={name ?? 'ellipse-outline'} size={size} color={color} />;
        },
      })}
    >
      <Tab.Screen name="Home" component={HomeScreen} options={{ title: '홈' }} />
      <Tab.Screen name="Report" component={ReportTabScreen} options={{ title: '리포트' }} />
      <Tab.Screen name="Records" component={InBodyStackNavigator} options={{ title: '기록' }} />
      <Tab.Screen name="Settings" component={SettingsStackNavigator} options={{ title: '설정' }} />
    </Tab.Navigator>
  );
}

// 설문은 성별·출생연도를 반드시 받으므로, 둘 중 하나라도 비어 있으면 설문 전으로 본다.
// 가족력은 '모름'이 null로 저장되어 미응답과 구별할 수 없어 판단에 쓰지 않는다.
function needsSurvey(me: UserProfile): boolean {
  const isEmpty = (v: unknown) => v === null || v === undefined;
  return isEmpty(me.gender) || isEmpty(me.birth_year);
}

function SignedInRoot() {
  const [status, setStatus] = useState<'checking' | 'survey' | 'done'>('checking');

  useEffect(() => {
    api
      .getMe()
      .then((me) => setStatus(needsSurvey(me) ? 'survey' : 'done'))
      .catch(() => setStatus('done')); // 프로필 조회 실패 시 앱 사용은 막지 않는다
  }, []);

  if (status === 'checking') return <SplashScreen />;
  if (status === 'survey') return <HealthSurveyScreen onDone={() => setStatus('done')} />;
  return <MainTabs />;
}

export default function RootNavigator() {
  const { session, loading } = useAuth();
  const [splashDone, setSplashDone] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setSplashDone(true), 2000);
    return () => clearTimeout(t);
  }, []);

  if (!splashDone || loading) {
    return <SplashScreen />;
  }

  return (
    <NavigationContainer>
      {session ? <SignedInRoot /> : <AuthStackNavigator />}
    </NavigationContainer>
  );
}
