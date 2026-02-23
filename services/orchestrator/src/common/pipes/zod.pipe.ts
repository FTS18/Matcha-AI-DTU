import {
  PipeTransform,
  Injectable,
  ArgumentMetadata,
  BadRequestException,
} from '@nestjs/common';
import { ZodSchema } from 'zod';

@Injectable()
export class ZodValidationPipe implements PipeTransform {
  constructor(private schema: ZodSchema) {}

  transform(value: unknown, metadata: ArgumentMetadata) {
    try {
      const parsedValue = this.schema.parse(value);
      return parsedValue;
    } catch (error: any) {
      if (error.errors) {
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
