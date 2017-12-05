curFolder=$(pwd)

# Run CBSContainer get its unique encrypt key, file UUID, VerifyKey
cd containers/createContainer/cbsContainer
echo >output.txt
docker run --rm --add-host dockerhost:192.168.65.1 -v $curFolder/containers/createContainer/cbsContainer/output.txt:/output.txt datasharing/cbs

# Run UMContainer its unique encrypt key, file UUID, VerifyKey
cd ../umContainer
echo >output.txt
docker run --rm --add-host dockerhost:192.168.65.1 -v $curFolder/containers/createContainer/umContainer/output.txt:/output.txt datasharing/um

cd ../../ttpImage
rm -R output/
mkdir output
# Commented for now, needs to be executed based on output of containers above, and implemented in input.txt
docker run --rm --add-host dockerhost:192.168.65.1 -v $curFolder/containers/ttpImage/output:/output -v $curFolder/containers/ttpImage/input.json:/input.txt datasharing/ttp
