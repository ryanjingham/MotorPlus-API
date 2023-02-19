FROM node:16

WORKDIR /MotorPlusAPI

COPY . .

RUN npm install

EXPOSE 3000

CMD [ "node", "app.js" ]