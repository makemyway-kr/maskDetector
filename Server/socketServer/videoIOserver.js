import { Server } from 'socket.io'
import http from 'http';
//비디오 중계용 js서버
const server = http.createServer().listen(5555);

var io = new Server(server)

//비디오를 받아서 서버에 넘겨줌.
io.sockets.on('connection', (socket) => {
    socket.join('video connection');
    //비디오 클라이언트로부터 비디오 도착.
    socket.on('videoIncoming', (videoData) => {
        console.log('video incoming');
        io.emit('videoProcess', videoData);
        //ML서버로 전송
    });
    //마스크 없음/턱스크에 따라서 클라이언트 웹으로 전송.
    socket.on('noMask', (resultData) => {
        console.log(resultData);
        io.sockets.emit('result', ['noMask' ,resultData]);
    })

    socket.on('partialMask',(resultData) =>{
        console.log(resultData);
        io.sockets.emit('result', ['partialMask' , resultData])
    })
});