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
import type { AuthStackParamList, InBodyStackParamList } from './types';

const TEAL = '#00C2A8';

const AuthStack = createNativeStackNavigator<AuthStackParamList>();
const Tab = createBottomTabNavigator();
const InBodyStack = createNativeStackNavigator<InBodyStackParamList>();

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
        options={{ title: '기록' }}
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
      <Tab.Screen name="Settings" component={SettingsScreen} options={{ title: '설정' }} />
    </Tab.Navigator>
  );
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
      {session ? <MainTabs /> : <AuthStackNavigator />}
    </NavigationContainer>
  );
}
