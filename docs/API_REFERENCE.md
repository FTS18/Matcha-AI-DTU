# REST API Reference

All requests to protected endpoints must include the `Authorization: Bearer <JWT>` header. The base path for these endpoints is `/api/v1`.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/auth/register` | Create a new user account |
| `POST` | `/auth/login` | Authenticate and receive a JWT |
| `GET` | `/auth/me` | Retrieve the authenticated user's profile |
| `GET` | `/matches` | List all matches for the logged-in user |
| `GET` | `/matches/:id` | Get details of a single match |
| `DELETE` | `/matches/:id` | Delete a match |
| `GET` | `/matches/stats` | Get match statistics for the user |
| `POST` | `/matches/:id/reanalyze` | Reanalyze a match or generate a reel (with `aspect_ratio` in body) |
| `POST` | `/matches/upload` | Upload a video for analysis (`multipart/form-data`) |
| `POST` | `/matches/youtube` | Upload a YouTube video for analysis |
| `GET` | `/matches/yt-info` | Get YouTube video info |
| `PATCH` | `/matches/:matchId/highlights/:highlightId` | Update a single highlight (timestamps, commentary, etc.) |
| `DELETE` | `/matches/:matchId/highlights/:highlightId` | Delete a single highlight |

## Examples

### `POST /auth/login`
Authenticate a user and receive a JWT access token.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJhbG...",
  "user": {
    "id": "user-uuid",
    "email": "user@example.com",
    "name": "John Doe"
  }
}
```

### `POST /matches/upload`
Upload a video file for analysis.

**Headers:**
```
Authorization: Bearer <JWT>
Content-Type: multipart/form-data
```

**Form Data:**
- `file`: (Binary Video File)

**Response (201 Created):**
```json
{
  "id": "match-uuid",
  "status": "UPLOADED",
  "uploadUrl": "http://localhost:4000/uploads/match.mp4",
  "progress": 0,
  "duration": 0,
  "createdAt": "2024-02-21T18:30:00.000Z"
}
```
