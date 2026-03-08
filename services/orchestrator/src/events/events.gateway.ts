import {
  WebSocketGateway,
  WebSocketServer,
  SubscribeMessage,
  MessageBody,
  ConnectedSocket,
} from '@nestjs/websockets';
import { Server, Socket } from 'socket.io';
import { WsEvents } from '@matcha/shared';

@WebSocketGateway({
  cors: {
    origin: true,
    credentials: true,
  },
})
export class EventsGateway {
  @WebSocketServer()
  server: Server;

  @SubscribeMessage(WsEvents.JOIN_MATCH)
  handleJoinMatch(
    @MessageBody() matchId: string,
    @ConnectedSocket() client: Socket,
  ) {
    client.join(matchId);
    return { event: 'joined', data: matchId };
  }
}
