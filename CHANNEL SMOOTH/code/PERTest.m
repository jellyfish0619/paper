function [numPacketErrors,numPkt]=PERTest(vht,txPSDU,tgacChannel,maxNumErrors,maxNumPackets,packetSNR,ind,fs)                
numPacketErrors=0;
numPkt=1;
seed=1;
a=zeros(242,1);
 while (numPacketErrors<=maxNumErrors && numPkt<=maxNumPackets)
                seed=seed+1;
                tx = wlanWaveformGenerator(txPSDU,vht);
%                 Add trailing zeros to allow for channel delay
                tx = [tx; zeros(50,vht.NumTransmitAntennas)]; %#ok<AGROW>
%                 Pass the waveform through the fading channel model
                reset(tgacChannel);
                rx = tgacChannel(tx);
                %reset(tgacChannel); % Reset channel for different realization
%                 Add noise
% 
                rng(seed);
                rx = awgn(rx,packetSNR);
          
%                 Allow same noise realization to be used subsequently
%                 Packet detect and determine coarse packet offset
                coarsePktOffset = wlanPacketDetect(rx,vht.ChannelBandwidth);
                if isempty(coarsePktOffset) % If empty no L-STF detected; packet error
                        numPacketErrors = numPacketErrors+1;
                        numPkt = numPkt+1;
                    continue; % Go to next loop iteration
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
                if pktOffset>90
                    numPacketErrors = numPacketErrors+1;
                    numPkt = numPkt+1;
                    continue; % Go to next loop iteration
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
                %chanEstSSPilots = vhtSingleStreamChannelEstimate(vhtltfDemod,vht);

                [~, chanEstSSPilots] = wlanVHTLTFChannelEstimate(vhtltfDemod, vht);
                
                chanEst = wlanVHTLTFChannelEstimate(vhtltfDemod,vht);
% %                 chanEst = vhtBeamformingRemoveCSD(chanEst , ...
% %                vht.ChannelBandwidth,vht.NumSpaceTimeStreams);%移除影响
%               Extrct VHT Data samples from the waveform

                for p =1:242
                 a(p)=trace(squeeze(chanEst(p,:,:))*squeeze(chanEst(p,:,:))');
                end
%                 plot(a,'Color','r');

               vhtdata = rx(pktOffset+(ind.VHTData(1):ind.VHTData(2)),:);
%                 vhtdata = rx(pktOffset+(ind.HEData(1):ind.HEData(2)),:);
% %                 Estimate the noise power in VHT data field
                nVarVHT = vhtNoiseEstimate(vhtdata,chanEstSSPilots,vht);
%                 Recover the transmitted PSDU in VHT Data
                rxPSDU = wlanVHTDataRecover(vhtdata,chanEst ,nVarVHT,vht,...
                    'LDPCDecodingMethod','layered-bp');
%                 Determine if any bits are in error, i.e. a packet error
                packetError = any(biterr(txPSDU,rxPSDU));
                numPacketErrors = numPacketErrors+packetError;
                numPkt = numPkt+1;
                
 end