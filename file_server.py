import asyncio
import websockets
import json
import os

from websockets.exceptions import ConnectionClosed


class FileServer:
    def __init__(self, host='localhost', port=5557, path=r"D:\veFox"):
        self.host = host
        self.port = port
        self.path = path

    async def handler(self, websocket):
        try:
            await websocket.send(json.dumps({
                'status': 'connected',
                'message': 'connect to server',
            }))

            async for message in websocket:
                await self.handle_message(message, websocket)

        except ConnectionClosed:
            print('Соединение разорвано')

    async def handle_message(self, message, websocket):
        try:
            data = json.loads(message)
            command = data.get('command')

            response = await self.process_command(command, data)

            await websocket.send(json.dumps(response))

        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                'error': 'Невалидные JSON-данные'
            }))

    async def process_command(self, command, data):
        if command == 'ping':
            return {'status': 'success', 'response': 'pong'}
        else:
            return {'status': 'error', 'response': 'Неизвестная команда'}

    def get_channels(self):
        channels = []

        for channel in os.listdir(self.path):
            check = self.check_channel(channel)

            if check:
                channels.append(channel)

        return channels

    # проверка что это действительно канал с папками видео
    def check_channel(self, channel_path):
        full_path = os.path.join(self.path, channel_path)

        videos = os.listdir(full_path)

        if not videos:
            return False

        if os.path.isdir(full_path):
            check_all_videos = all(
                os.path.isdir(os.path.join(full_path, video))
                for video in videos
            )
            return check_all_videos
        else:
            return False

    # получаем все видео
    def get_all_videos(self):
        all_videos = []

        channels = self.get_channels()

        if not channels:
            return []

        for channel in channels:
            check = self.check_channel(channel)

            if check:
                videos = os.listdir(os.path.join(self.path, channel))

                if videos:
                    all_videos.extend(videos)

        print(all_videos)


    async def start_server(self):
        print(f'Запуск сервера на ws://{self.host}:{self.port}')

        async with websockets.serve(self.handler, self.host, self.port):
            print('Сервер запущен')

            await asyncio.Future()

    def start(self):
        self.get_all_videos()
        asyncio.run(self.start_server())


if __name__ == "__main__":
    server = FileServer()
    server.start()
