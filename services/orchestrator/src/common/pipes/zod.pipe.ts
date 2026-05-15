import { Injectable, PipeTransform, BadRequestException } from '@nestjs/common';
import { ZodSchema, ZodError } from 'zod';

@Injectable()
export class ZodValidationPipe implements PipeTransform {
  constructor(private schema: ZodSchema) {}

  transform(value: unknown) {
    try {
      return this.schema.parse(value) as unknown;
    } catch (error: unknown) {
      if (error instanceof ZodError) {
        console.log(
          '[ZOD] Validation failed:',
          JSON.stringify(error.errors, null, 2),
        );
        throw new BadRequestException({
          message: 'Validation failed',
          errors: error.errors,
        });
      }
      console.log('[ZOD] Validation failed (no details)');
      throw new BadRequestException('Validation failed');
    }
  }
}
