import { createApiClient } from "@matcha/shared";
import { env } from "./env";

// createApiClient expects the base orchestrator URL (e.g. http://localhost:4000)
// NEXT_PUBLIC_API_URL usually includes /api/v1
const orchestratorBase = env.NEXT_PUBLIC_API_URL.replace("/api/v1", "");
export const client = createApiClient(orchestratorBase);
