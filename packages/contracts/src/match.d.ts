import { z } from "zod";
export declare const CreateMatchSchema: z.ZodObject<
  {
    date: z.ZodString;
    duration: z.ZodNumber;
    location: z.ZodString;
    title: z.ZodString;
  },
  "strip",
  z.ZodTypeAny,
  {
    title: string;
    date: string;
    duration: number;
    location: string;
  },
  {
    title: string;
    date: string;
    duration: number;
    location: string;
  }
>;
export type CreateMatchInput = z.infer<typeof CreateMatchSchema>;
//# sourceMappingURL=match.d.ts.map
