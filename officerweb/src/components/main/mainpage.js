import React, { useEffect } from "react";
import styled from "styled-components";
import { useState } from "react";
import socketIoClient from 'socket.io-client';
import { Buffer } from 'buffer';
import './main.css';

const socketServer = 'http://15.164.111.113:5555'; //서버 주소

const socket = socketIoClient.connect(socketServer,{ transports: ['websocket', 'polling', 'flashsocket'] });

const PageBody = styled.div`
    width : 100%;
    justify-content : center;
    text-align : center;
`

const ImageBox = styled.div`
    width : 80%;
    height : 60%;
    justify-content: center;
    object-fit : cover;    
`

const MainPage = () => {
    const [image, setImage] = useState(['','']);
    useEffect(() => {
        socket.on('connection' , sock => {
            console.log('connected');
        });
    }, []);

    useEffect(() => {
        socket.on('result', (data) => {
            console.log('image comming');
            console.log(data);
            setImage([data[0],Buffer.from(data[1], "base64").toString()]);
        })
    })

    return (<PageBody >
        <div className="title">마스크 미착용자 알림 프로그램</div>
        <div className="type">{image[0]}</div>
        <ImageBox>{ image[1] !== '' &&<img class = "maskImg"src={"data:image/jpeg;base64," + image[1]} />}</ImageBox>
    </PageBody>
    )
}

export default MainPage;