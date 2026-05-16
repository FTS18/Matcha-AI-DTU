import { z } from 'zod';

const envSchema = z.object({
  NODE_ENV: z
    .enum(['development', 'production', 'test', 'provision'])
    .default('development'),
  PORT: z.coerce.number().default(4000),
  DATABASE_URL: z.string().url(),
  JWT_SECRET: z.string().min(1),
  CORS_ORIGIN: z.string().default('http://localhost:3000'),
  INFERENCE_URL: z.string().url().default('http://localhost:8000'),
  REDIS_HOST: z.string().default('localhost'),
  REDIS_PORT: z.coerce.number().default(6379),
  HF_TOKEN: z.string().optional(),
});

export function validate(config: Record<string, unknown>) {
  const result = envSchema.safeParse(config);

  if (!result.success) {
    console.error('âŒ Invalid environment variables:');
    console.error(JSON.stringify(result.error.flatten().fieldErrors, null, 2));
    throw new Error('Invalid environment variables');
  }

  return result.data;
}
