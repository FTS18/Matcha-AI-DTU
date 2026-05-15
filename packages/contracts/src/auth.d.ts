import { z } from "zod";
export declare const LoginSchema: z.ZodObject<
  {
    email: z.ZodString;
    password: z.ZodString;
  },
  "strip",
  z.ZodTypeAny,
  {
    email: string;
    password: string;
  },
  {
    email: string;
    password: string;
  }
>;
export declare const RegisterSchema: z.ZodObject<
  {
    email: z.ZodString;
    password: z.ZodString;
    firstName: z.ZodString;
    lastName: z.ZodString;
  },
  "strip",
  z.ZodTypeAny,
  {
    email: string;
    password: string;
    firstName: string;
    lastName: string;
  },
  {
    email: string;
    password: string;
    firstName: string;
    lastName: string;
  }
>;
export type LoginInput = z.infer<typeof LoginSchema>;
export type RegisterInput = z.infer<typeof RegisterSchema>;
//# sourceMappingURL=auth.d.ts.map
