import 'react-native-url-polyfill/auto';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = 'https://qehwljzduzblygpzitew.supabase.co';
const supabaseAnonKey =
  'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFlaHdsanpkdXpibHlncHppdGV3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzg2NDAyOTAsImV4cCI6MjA5NDIxNjI5MH0.WTeA2GKSiAhpuVrc_oMBIYZ7s23OPSm4OBU2qhL6Asw';

export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    storage: AsyncStorage,
    autoRefreshToken: true,
    persistSession: true,
    detectSessionInUrl: false,
  },
});
