function [numPacketerrors,numPkt]=PERTest(vht,txPSDU,tgacChannel,maxNumErrors,maxNumPackets,packetSNR,ind,fs)                
numPacketErrors=0;
proba=cell(1,maxNumPackets);
numpkt=cell(1,maxNumPackets);
parfor i=1:maxNumPackets

                tx = wlanWaveformGenerator(txPSDU,vht);
%                 Add trailing zeros to allow for channel delay
                tx = [tx; zeros(50,vht.NumTransmitAntennas)]; %#ok<AGROW>
%                 Pass the waveform through the fading channel model
                reset(tgacChannel);
                rx = tgacChannel(tx);
                %reset(tgacChannel); % Reset channel for different realization
%                 Add noise
% 
                %rng(seed);
                rx = awgn(rx,packetSNR);
          
%                 Allow same noise realization to be used subsequently
%                 Packet detect and determine coarse packet offset
                coarsePktOffset = wlanPacketDetect(rx,vht.ChannelBandwidth);
                if isempty(coarsePktOffset) % If empty no L-STF detected; packet error
                        proba{i}=1;
                        numpkt{i}=1;
                    continue
                end
%                 Extract L-STF and perform coarse frequency offset correction
                lstf = rx(coarsePktOffset+(ind.LSTF(1):ind.LSTF(2)),:);
                coarseFreqOff = wlanCoarseCFOEstimate(lstf,vht.ChannelBandwidth);
                rx = frequencyOffset(rx,fs,-coarseFreqOff);
%                 Extract the non-HT fields and determine fine packet offset
                nonhtfields = rx(coarsePktOffset+(ind.LSTF(1):ind.LSIG(2)),:);
                finePktOffset = wlanSymbolTimingEstimate(nonhtfields,...
                        vht.ChannelBandwidth);
%                 Determine final packet offset
                pktOffset = coarsePktOffset+finePktOffset;
%                 If packet detected outwith the range of expected delays from the
%                 channel modeling; packet error
                if pktOffset>50
                        proba{i}=1;
                        numpkt{i}=1;
                    continue
                end
%                 Extract L-LTF and perform fine frequency offset correction
                lltf = rx(pktOffset+(ind.LLTF(1):ind.LLTF(2)),:);
                fineFreqOff = wlanFineCFOEstimate(lltf,vht.ChannelBandwidth);
                rx = frequencyOffset(rx,fs,-fineFreqOff);
%                 Extract VHT-LTF samples from the waveform, demodulate and perform
%                 channel estimation
                vhtltf = rx(pktOffset+(ind.VHTLTF(1):ind.VHTLTF(2)),:);
                vhtltfDemod = wlanVHTLTFDemodulate(vhtltf,vht);
%                 Channel estimate
                chanEstSSPilots = vhtSingleStreamChannelEstimate(vhtltfDemod,vht);
                chanEst = wlanVHTLTFChannelEstimate(vhtltfDemod,vht);
%                 chanEst = vhtBeamformingRemoveCSD(chanEst , ...
%                vht.ChannelBandwidth,vht.NumSpaceTimeStreams);%移除影响
%               Extrct VHT Data samples from the waveform


                vhtdata = rx(pktOffset+(ind.VHTData(1):ind.VHTData(2)),:);
%                 Estimate the noise power in VHT data field
                nVarVHT = vhtNoiseEstimate(vhtdata,chanEstSSPilots,vht);
%                 Recover the transmitted PSDU in VHT Data
                rxPSDU = wlanVHTDataRecover(vhtdata,chanEst ,nVarVHT,vht,...
                    'LDPCDecodingMethod','layered-bp');
%                 Determine if any bits are in error, i.e. a packet error
                proba{i} = any(biterr(txPSDU,rxPSDU));
                numpkt{i}=1;
end
numPacketerrors=proba{maxNumPackets};
numPkt=numpkt{maxNumPackets};
% numPacketerrors=length(find(packetError==1));
% numPkt=length(find(numpkt==1));