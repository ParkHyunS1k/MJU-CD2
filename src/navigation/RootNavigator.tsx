import React from 'react';
import { ActivityIndicator, View } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';

import { useAuth } from '../contexts/AuthContext';
import LoginScreen from '../screens/LoginScreen';
import HomeScreen from '../screens/HomeScreen';
import InBodyListScreen from '../screens/InBodyListScreen';
import InBodyCreateScreen from '../screens/InBodyCreateScreen';
import InBodyDetailScreen from '../screens/InBodyDetailScreen';
import InBodyUploadScreen from '../screens/InBodyUploadScreen';
import InBodyReviewScreen from '../screens/InBodyReviewScreen';
import AnalysisScreen from '../screens/AnalysisScreen';
import type { InBodyStackParamList } from './types';

const Tab = createBottomTabNavigator();
const InBodyStack = createNativeStackNavigator<InBodyStackParamList>();

function InBodyStackNavigator() {
  return (
    <InBodyStack.Navigator>
      <InBodyStack.Screen
        name="InBodyList"
        component={InBodyListScreen}
        options={{ title: 'InBody 기록' }}
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
        options={{ title: '결과지 업로드', presentation: 'modal' }}
      />
      <InBodyStack.Screen
        name="InBodyReview"
        component={InBodyReviewScreen}
        options={{ title: '추출 결과 확인' }}
      />
    </InBodyStack.Navigator>
  );
}

function MainTabs() {
  return (
    <Tab.Navigator>
      <Tab.Screen name="Home" component={HomeScreen} options={{ title: '홈' }} />
      <Tab.Screen
        name="InBody"
        component={InBodyStackNavigator}
        options={{ title: 'InBody', headerShown: false }}
      />
      <Tab.Screen
        name="Analysis"
        component={AnalysisScreen}
        options={{ title: 'AI 분석' }}
      />
    </Tab.Navigator>
  );
}

export default function RootNavigator() {
  const { session, loading } = useAuth();

  if (loading) {
    return (
      <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
        <ActivityIndicator size="large" />
      </View>
    );
  }

  return (
    <NavigationContainer>
      {session ? <MainTabs /> : <LoginScreen />}
    </NavigationContainer>
  );
}
