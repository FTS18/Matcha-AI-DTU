import { createEnv } from "@t3-oss/env-core";
import { z } from "zod";

export const env = createEnv({
 server: {
 NODE_ENV: z
 .enum(["development", "test", "production"])
 .default("development"),
 DATABASE_URL: z.string().url(),
 PORT: z.string().default("4000"),
 },
 clientPrefix: "NEXT_PUBLIC_",
 client: {},
 runtimeEnv: process.env,
 emptyStringAsUndefined: true,
});
