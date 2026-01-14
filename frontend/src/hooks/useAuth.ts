import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useRouter } from 'next/navigation';
import { LoginCredentials, RegisterCredentials } from '@/src/types';
import { authApi } from '../lib/api/auth';
import { useGlobalContext } from '../context/GlobalContext';

export function useAuth() {
  const queryClient = useQueryClient();
  const router = useRouter();
  const {setIsSignUpMode} = useGlobalContext();

  // 1. User State (The "Session")
  const { data: user, isLoading: isLoadingUser, isError } = useQuery({
    queryKey: ['auth', 'me'],
    queryFn: authApi.getMe,
    retry: false, // Don't retry if 401
    staleTime: 1000 * 60 * 5, // Cache user data for 5 mins
  });

  // 2. Login Mutation
  const loginMutation = useMutation({
    mutationFn: (creds: LoginCredentials) => authApi.login(creds),
    onSuccess: (data) => {
      // Store token
      localStorage.setItem('token', data.access_token);
      
      // Invalidate 'me' query so it refetches user data immediately
      queryClient.invalidateQueries({ queryKey: ['auth', 'me'] });
      
      // Redirect to dashboard
      router.push('/');
    },
  });

  // 3. Register Mutation
  const registerMutation = useMutation({
    mutationFn: (creds: RegisterCredentials) => authApi.register(creds),
    onSuccess: () => {
      // Auto-login logic could go here, or just switch to login view
      setIsSignUpMode(false);
    },
  });

  // 4. Logout Mutation
  const logoutMutation = useMutation({
    mutationFn: authApi.logout,
    onSettled: () => {
      // Always remove token and clear cache, even if API fails
      localStorage.removeItem('token');
      queryClient.setQueryData(['auth', 'me'], null);
      router.push('/');
    },
  });

  return {
    user,
    isLoadingUser,
    isAuthenticated: !!user,
    login: loginMutation.mutate,
    isLoggingIn: loginMutation.isPending,
    loginError: loginMutation.error,
    register: registerMutation.mutate,
    isRegistering: registerMutation.isPending,
    registerError: registerMutation.error,
    logout: logoutMutation.mutate,
  };
}