FROM node:14

WORKDIR /MotorPlusAPI

COPY . .

RUN npm install

EXPOSE 3000

CMD [ "npm", "start" ]