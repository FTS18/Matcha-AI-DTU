import {
  Controller,
  Post,
  Body,
  Req,
  UseGuards,
  Get,
  UnauthorizedException,
  UsePipes,
} from '@nestjs/common';
import { AuthService } from './auth.service';
import { JwtAuthGuard } from './jwt-auth.guard';
import { ZodValidationPipe } from '../common/pipes/zod.pipe';
import { LoginSchema, RegisterSchema } from '@matcha/contracts';
import type { LoginInput, RegisterInput } from '@matcha/shared';
import { User } from '@matcha/database';

interface AuthRequest extends Express.Request {
  user: {
    userId: string;
    email: string;
  };
}

@Controller('auth')
export class AuthController {
  constructor(private authService: AuthService) {}

  @Post('login')
  @UsePipes(new ZodValidationPipe(LoginSchema))
  async login(@Body() body: LoginInput) {
    const user = await this.authService.validateUser(body.email, body.password);
    if (!user) {
      throw new UnauthorizedException('Invalid credentials');
    }
    return this.authService.login(user as User);
  }

  @Post('register')
  @UsePipes(new ZodValidationPipe(RegisterSchema))
  async register(@Body() body: RegisterInput) {
    const user = await this.authService.register(body);
    return this.authService.login(user as User); // auto-login after register
  }

  @UseGuards(JwtAuthGuard)
  @Get('me')
  async getProfile(@Req() req: AuthRequest) {
    const userProfile = await this.authService.getUserById(req.user.userId);
    if (!userProfile) {
      throw new UnauthorizedException();
    }
    return {
      id: userProfile.id,
      email: userProfile.email,
      name:
        userProfile.firstName && userProfile.lastName
          ? `${userProfile.firstName} ${userProfile.lastName}`
          : userProfile.firstName || userProfile.email.split('@')[0],
    };
  }
}
