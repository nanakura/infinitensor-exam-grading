import { ProChat } from '@ant-design/pro-chat';

import { useTheme } from 'antd-style';

import axios from 'axios'

const request = axios.create({
  baseURL: 'http://127.0.0.1:8080/',
  timeout: 36000,
  headers: { 'Content-Type': 'application/json' }
})

function App() {
  const theme = useTheme();

  const url = new URL("/api/init/1111", "http://localhost:8080");
  const es = new EventSource(url);

  return (
    <>
      <div className='flex flex-col w-full h-screen' style={{ background: theme.colorBgLayout }}>
        <ProChat
          className='flex-1'
          helloMessage={
            '欢迎使用 ProChat ，我是你的专属机器人'
          }
          request={async (messages) => {
            console.log(messages);

            // 发送到服务器
            const userContent = messages.filter(it => it.role === 'user')
            const message = userContent[userContent.length - 1].content
            console.log(message)

            const stream = new ReadableStream({
              async start(controller) {
                es.onmessage = (e) => {
                  const data = e.data || "";
                  if (data === "|DONE|") {
                    es.close();
                    controller.close();
                    return;
                  }

                  try {
                    if (!!data) {
                      console.log(data)
                      const encoder = new TextEncoder();
                      controller.enqueue(encoder.encode(data));
                    }
                  } catch (error) {
                    console.error('Error  parsing data:', error);
                    es.close();
                    controller.close();
                  }
                };

                es.onerror = () => {
                  console.error('EventSource  connection error');
                  es.close();
                  controller.close();
                };
              },
              cancel() {
                es.close();
              },
            });
            request.post('/api/chat', {
              session_id: '1111',
              message: message
            });
            return new Response(stream);
          }}
        />
      </div>
    </>
  )
}

export default App
