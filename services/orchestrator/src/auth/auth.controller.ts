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

@Controller('auth')
export class AuthController {
  constructor(private authService: AuthService) {}

  @Post('login')
  @UsePipes(new ZodValidationPipe(LoginSchema))
  async login(@Body() body: any) {
    const user = await this.authService.validateUser(body.email, body.password);
    if (!user) {
      throw new UnauthorizedException('Invalid credentials');
    }
    return this.authService.login(user);
  }

  @Post('register')
  @UsePipes(new ZodValidationPipe(RegisterSchema))
  async register(@Body() body: any) {
    const user = await this.authService.register(body);
    return this.authService.login(user); // auto-login after register
  }

  @UseGuards(JwtAuthGuard)
  @Get('me')
  async getProfile(@Req() req: any) {
    const user = await this.authService.getUserById(req.user.userId);
    if (!user) {
      throw new UnauthorizedException();
    }
    return {
      id: user.id,
      email: user.email,
      name:
        user.firstName && user.lastName
          ? `${user.firstName} ${user.lastName}`
          : user.name || user.email.split('@')[0],
    };
  }
}
