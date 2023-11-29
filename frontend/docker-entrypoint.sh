#!/bin/sh

ln -s /usr/src/node_modules node_modules

if [ ${NG_CMD} == SERVE ]
then
   echo "Starting Angular development Server"
   ng serve --host=0.0.0.0 --serve-path=/admin/frontend/ --base-href=/admin/frontend/ --disable-host-check
elif [ ${NG_CMD} == BUILD ]
then
   echo "Started building Angular dist"
   rm -rf /frontend/dist
   chown -R root:root .
   source ~/.bashrc
   npm install
   ng build --configuration production
   chown -R ${HOST_USER_ID}:${HOST_GROUP_ID} dist
   echo "Angular build process completed."
elif [ ${NG_CMD} == SERVE_DIST ]
then
   echo "Started serving Angular from dist"
elif [ ${NG_CMD} == TEST ]
then
   echo "Starting Angular tests"
   ng test
else
   echo "ALLOWED NG_CMD [SERVE | BUILD | SERVE_DIST | TEST] - Serving Angular from dist"
fi
