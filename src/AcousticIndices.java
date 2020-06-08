import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.*;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayDeque;
import java.util.Arrays;

/**
 * Created by aalle on 25/07/2017.
 */
public class AcousticIndices {
    static final double[] RAIN_PSD={1.206376, 0.867175,1.376895,0.824139,0.116983};
    public static double[] powerSpectralDensity(Complex[][] stft,int numTransforms,int windowSize)
    {
        ////System.out.println(numTransforms);
        double[] powerSpectralDensities= new double[windowSize/2];
        double hammingSquareSum=HammingWindow.getHammingSquareSum(windowSize);
        for(int i=0;i<windowSize/2;i++)
        {
            double bandSum=0;
            for(int j=0;j<numTransforms;j++)
            {
                ////System.out.println("j "+j+" i "+i);
                bandSum+=/*hammingSquareSum**/Math.pow(complexMod(stft[j][i]),2);
            }
            powerSpectralDensities[i]=bandSum/numTransforms;
            ////System.out.println("PSD "+i+": "+powerSpectralDensities[i]);
        }
        return powerSpectralDensities;
    }

    public static double frequencySubset(double[] spectrum,double[] frequencies, int min,int max)
    {
        ////System.out.println(min+"-"+max);
        int numBands=0;
        double sumFrequencies=0;
        for(int i=0;i<frequencies.length;i++)
        {
            if(frequencies[i]>min&&frequencies[i]<max)
            {
                ////System.out.println(frequencies[i]+": "+spectrum[i]);
                sumFrequencies+=spectrum[i];
                numBands++;
            }
        }
        ////System.out.println("sumFrequencies: "+sumFrequencies+" numBands "+numBands);
        return sumFrequencies/numBands;
    }

    public static double signalToNoise(Complex[][] stft,int numTransforms,double[] frequencies,double meanPSD, int windowSize, int min,int max)
    {
        //System.out.println("STFT Length: "+stft.length);
        //System.out.println("numTransfomrs: "+numTransforms);
        double[] bandAverage=new double[numTransforms];
        double hammingSquareSum=HammingWindow.getHammingSquareSum(windowSize);
        for(int i=0;i<numTransforms;i++) {
            int numBands=0;
            double sumFrequencies=0;
            for (int j = 0; j < frequencies.length; j++) {
                if (frequencies[j] > min && frequencies[j] < max) {
                    ////System.out.println(frequencies[j]+": "+Math.pow(complexMod(stft[i][j]),2));
                    sumFrequencies += /*hammingSquareSum**/Math.pow(complexMod(stft[i][j]),2);
                    numBands++;
                }
            }
            bandAverage[i]=sumFrequencies/numBands;
        }
        double sumDiff=0;
        int numSamples=0;
        for(int i=0;i<bandAverage.length;i++)
        {
            ////System.out.println(bandAverage[i]+" "+meanPSD);
            sumDiff+=Math.pow(bandAverage[i]-meanPSD,2);
        }
        //System.out.println("meanPSD: "+meanPSD+" bandAverage.length "+bandAverage.length+" sumDiff "+sumDiff);
        ////System.out.println(Math.sqrt(((double)1/(bandAverage.length-1))*sumDiff));
        if(meanPSD==0)
            return 0;
        else
            return Math.pow(sumDiff/(numTransforms-1),0.5)/(meanPSD);
    }

    public static double freqSNR(double[] frequencies,double[] psdSpectrum, double meanPSD,int min,int max)
    {
        double sumPSDDiffs=0;
        int numBands=0;
        for(int i=0;i<psdSpectrum.length;i++)
        {
            if(frequencies[i]>min&&frequencies[i]<max)
            {
                //System.out.println(meanPSD+" "+psdSpectrum[i]);
                sumPSDDiffs+=Math.pow(meanPSD-psdSpectrum[i],2);
                numBands++;
            }
        }
        //System.out.println(meanPSD);
        //System.out.println(sumPSDDiffs);
        //System.out.println(numBands);
        //System.out.println(meanPSD/Math.pow((((double)1/(numBands-1))*sumPSDDiffs),0.5));
        return meanPSD/(Math.pow((((double)1/(numBands-1))*sumPSDDiffs),0.5));
    }
    public static double segmentalSNR(Complex[][]stft,double[] frequencies,int min,int max,double segmentLength,double sampleRate)
    {
        int stftLength=stft.length;
        int windowSize=stft[0].length;
        int segmentInSamples=(int)(segmentLength*sampleRate);
        ////System.out.println("segmentInSamples: "+segmentInSamples);
        int segmentInWindows=segmentInSamples/windowSize;
        int totalSegments=stftLength/segmentInWindows;
        double segSNR=0;
        for(int i=0;i<totalSegments;i++)
        {
            Complex[][] shortSTFT = new Complex[segmentInWindows][windowSize];
            for(int j=0;j<segmentInWindows;j++) {
                for (int k = 0; k < windowSize; k++) {
                    shortSTFT[j][k] = stft[i*segmentInWindows+j][k];
                }
            }
            double[] psdSepctrum=powerSpectralDensity(shortSTFT,shortSTFT.length,windowSize);
            double meanPSD=frequencySubset(psdSepctrum,frequencies,min,max);
            segSNR+=signalToNoise(shortSTFT,shortSTFT.length,frequencies,meanPSD,windowSize,min,max)/totalSegments;
        }
        return segSNR;
    }

    public static double complexMod(Complex value) {
        return Math.sqrt(Math.pow(value.getReal(),2)+Math.pow(value.getImaginary(),2));
    }
    public static double[] samplesToDB(short[] samples)
    {
        double[] samplesToDB=new double[samples.length];
        for(int i=0;i<samples.length;i++)
        {
            samplesToDB[i]=20*Math.log((double)samples[i]/Math.pow(2,15));
        }
        return samplesToDB;
    }
    public static double[] normaliseSample(short[] samples,int bitRate)
    {
        double[] normSamples=new double[samples.length];
        for(int i=0;i<samples.length;i++)
        {
            normSamples[i]=(double)samples[i]/Math.pow(2,(bitRate-1));
        }
        return normSamples;
    }
    public static double[] getFreqeuencies(int windowSize,int sampleRate)
    {
        double[] frequencies=new double[windowSize/2];
        for(int i=0;i<windowSize/2;i++) {
            frequencies[i] = i * sampleRate / windowSize;
        }
        return frequencies;
    }
    public static Complex[][] stft(File file,int windowSize) {
        System.out.println("Starting FFT");
        FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.STANDARD);
            //double[] values=HammingWindow.toHammingWindow(WaveformNoiseRemoval.getNoDBValues(file),windowSize);
            double[] values = getNoDBValues(file);
            short[] finalValues = new short[values.length + windowSize - values.length % windowSize + windowSize / 2];
            ////System.out.println("Values length: " + values.length + " final values length: " + finalValues.length);
            long valueLength = values.length;
            ////System.out.println("ValueLength: "+finalValues.length);
            int totalFFTs = (int) valueLength / windowSize * 2 + 2;
            System.out.println("Total FFTs "+totalFFTs);
            ////System.out.println(valueLength%windowSize);
            if(valueLength%windowSize<(windowSize/2))
                totalFFTs--;
            int currentFrameNo = 0;
            int FFTsDone = 0;
            Complex[][] transforms = new Complex[totalFFTs][windowSize];
            //System.out.println("Transforms length "+transforms.length);
            while (valueLength > currentFrameNo) {
                double[] newValues = new double[windowSize];
                int thisSet = 0;
                while (thisSet < windowSize && currentFrameNo < valueLength) {
                    newValues[thisSet] = values[currentFrameNo];
                    thisSet++;
                    currentFrameNo++;
                }
                ////System.out.println("Last window "+currentFrameNo+", "+valueLength);
                if (currentFrameNo >= valueLength) {
                    ////System.out.println("Last window " + currentFrameNo + ", " + valueLength);
                    for (int i = thisSet; i < windowSize; i++) {
                        newValues[i] = 0;
                        currentFrameNo++;
                    }
                }
                /*if(FFTsDone<2)
                {
                    for(int i=0;i<newValues.length;i++)
                    {
                        //System.out.println(i+" "+newValues[i]);
                    }
                }*/
                newValues = HammingWindow.toHammingWindow(newValues, windowSize);
                /*if(FFTsDone<2)
                {
                    for(int i=0;i<newValues.length;i++)
                    {
                        //System.out.println(i+" "+newValues[i]);
                    }
                }*/
                transforms[FFTsDone] = fft.transform(newValues, TransformType.FORWARD);
                if(FFTsDone%10000==0)
                    System.out.println("FFTs done: "+FFTsDone);
                currentFrameNo-=windowSize/2;
                FFTsDone++;
            }
            //System.out.println("Transforms length 2: "+transforms.length);
        return transforms;
    }

    public static double[] iStft(Complex[][]stft) {
        FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.STANDARD);
        //double[] values=HammingWindow.toHammingWindow(WaveformNoiseRemoval.getNoDBValues(file),windowSize);
        double[] values=new double[stft.length*stft[0].length/2+stft[0].length];
        //System.out.println(values.length);
        for(int i=0;i<stft.length;i++)
        {
            Complex[] iFFT=fft.transform(stft[i],TransformType.INVERSE);
            double[] realiFFT=new double[iFFT.length];
            for(int j=0;j<iFFT.length;j++)
            {
                realiFFT[j]=iFFT[j].getReal();
            }
            for(int j=0;j<realiFFT.length;j++)
            {
                values[i*stft[0].length/2+j]+=realiFFT[j];
            }
        }
        return values;
    }

    public static Complex[][] fftFraction(Complex[][] fft,int part,int totalPart) {
        int newLength=fft.length/totalPart;
        Complex[][] fftFraction=new Complex[newLength][fft[0].length];
        int start=part*newLength;
        //System.out.println("Part: "+part);
        //System.out.println("FFT Length: "+fft.length);
        //System.out.println("newLength: "+newLength);
        //System.out.println("Start: "+start);
        for(int i=start;(i<start+newLength)&&(i<fft.length);i++)
        {
            for(int j=0;j<fft[0].length;j++)
            {
                fftFraction[i-start][j]=fft[i][j];
            }
        }
        return fftFraction;
    }

    public static Complex[] getAnalyticSignal(Complex[][]stft) {
        Complex[][] newSTFT=new Complex[stft.length][stft[0].length];
        for(int i=0;i<stft.length;i++)
        {
            for(int j=0;j<stft[0].length;j++)
            {
                newSTFT[i][j]=stft[i][j];
            }
        }
        int fileLength=newSTFT.length*newSTFT[0].length/2+newSTFT[0].length;
        ////System.out.println("FileLength: "+fileLength);
        Complex[] analyticSignal=new Complex[fileLength];
        int currentSample=0;
        int windowSize=newSTFT[0].length;
        FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.STANDARD);
        for(int i=0;i<newSTFT.length;i++)
        {
            for(int j=0;j<windowSize;j++)
            {
                if(j<windowSize/2)
                {
                    newSTFT[i][j]=newSTFT[i][j].multiply(2);
                }
                else if(j>windowSize/2)
                {
                    newSTFT[i][j]=new Complex(0,0);
                }
            }
            Complex[] windowSignal=fft.transform(newSTFT[i],TransformType.INVERSE);
            for(int k=0;k<windowSize;k++)
            {
                if(analyticSignal[currentSample]==null)
                    analyticSignal[currentSample]=windowSignal[k];
                else
                    analyticSignal[currentSample]=analyticSignal[currentSample].add(windowSignal[k]);
                /*if(currentSample%100==0)
                    System.out.println((double)currentSample/22050+": "+analyticSignal[currentSample]);*/
                currentSample++;
            }
            currentSample-=windowSize/2;
            ////System.out.println(currentSample);
        }
        for(int l=currentSample;l<analyticSignal.length;l++)
        {
            ////System.out.println(l);
            analyticSignal[l]=new Complex(0,0);
        }
        return analyticSignal;
    }
    public static double temporalEntropy(Complex[][] stft)
    {
        Complex[] analyticSignal=getAnalyticSignal(stft);
        double[] pmf=pmf(analyticSignal);
        double sumTemporalEntropy=0;
        for(int i=0;i<pmf.length;i++)
        {
            if(pmf[i]>0)
                sumTemporalEntropy-=pmf[i]*Math.log(pmf[i])/(Math.log(pmf.length));
        }
        return sumTemporalEntropy;
    }
    public static double spectralEntropy(Complex[][] stft)
    {
        double[] spectralEntropy=spectralSentropySpectrum(stft);
        double sumSpectralEntropy=0;
        for(int i=0;i<spectralEntropy.length;i++)
        {
            ////System.out.println(spectralEntropy[i]);
            sumSpectralEntropy+=spectralEntropy[i];
        }
        return (double)(sumSpectralEntropy/spectralEntropy.length);
    }
    public static double[] averagePMF(double[][]pmf)
    {
        double[] averagePMF=new double[pmf[0].length];
        for(int i=0;i<pmf.length;i++)
        {
            for(int j=0;j<pmf[0].length;j++)
            {
                averagePMF[j]+=pmf[i][j]/pmf.length;
            }
        }
        return averagePMF;
    }
    public static double[] pmfStdDev(double[][] pmf,double[] meanPMF) {
        double[] stdDevPMF=new double[meanPMF.length];
        double[] sumDiffPMF=new double[meanPMF.length];
        for(int i=0;i<pmf[0].length;i++)
        {
            for(int j=0;j<pmf.length;j++)
            {
                sumDiffPMF[i]+=Math.pow(Math.abs(meanPMF[i]-pmf[j][i]),2);
            }
            stdDevPMF[i]=Math.sqrt(sumDiffPMF[i]/pmf.length)/meanPMF[i]*100;
        }
        return stdDevPMF;
    }
    public static double[] spectralSentropySpectrum(Complex[][] stft)
    {
        double[][]pmf=pmf(stft);
        double[] spectralEntropy=new double[pmf.length];
        double sumSpectralEntropy=0;
        for(int i=0;i<pmf.length;i++)
        {
            for(int j=0;j<pmf[i].length;j++)
            {
                if(pmf[i][j]!=0)
                    spectralEntropy[i]-=pmf[i][j]*Math.log(pmf[i][j])/(Math.log(pmf[i].length));
            }
        }
        return spectralEntropy;
    }

    public static double spectralEntropySubset(Complex[][] stft,double[] frequencies,int min,int max)
    {
        ////System.out.println("SpectralEntropySubset "+min+"-"+max);
        double[][]pmf=pmfSubset(stft,frequencies,min,max);
        double[] spectralEntropy=new double[pmf.length];
        double sumSpectralEntropy=0;
        for(int i=0;i<pmf.length;i++)
        {
            for(int j=0;j<pmf[i].length;j++)
            {
                if(pmf[i][j]!=0) {
                    spectralEntropy[i] -= pmf[i][j] * Math.log(pmf[i][j]) / (Math.log(pmf[i].length));
                    ////System.out.println(spectralEntropy[i]);
                    sumSpectralEntropy+=spectralEntropy[i];
                }
            }
        }
        ////System.out.println(sumSpectralEntropy/spectralEntropy.length);
        return sumSpectralEntropy/spectralEntropy.length;
    }

    public static double[][] pmf(Complex[][] stft) {
        double[][] pmf=new double[stft.length][stft[0].length/2];
        for (int i = 0; i < stft.length; i++)
        {
            double[] spectrum=new double[stft[0].length/2];
            double sumSpectrum=0;
            for(int j=0;j<stft[i].length/2;j++)
            {
                spectrum[j]=complexMod(stft[i][j]);
                sumSpectrum+=spectrum[j];
            }
            for(int j=0;j<stft[i].length/2;j++)
            {
                if(sumSpectrum>0)
                    pmf[i][j]=spectrum[j]/sumSpectrum;
                else
                    pmf[i][j]=0;
            }
            pmf[i]=normaliseSpectrum(pmf[i]);
        }
        return pmf;
    }
    public static double [][] pmfSubset(Complex[][] stft,double[] frequencies,int min,int max)
    {
        //System.out.println("PMF Subset: "+min+"-"+max);
        int arrayLength=0;
        for(int i=0;i<frequencies.length;i++)
        {
            if(frequencies[i]>min&&frequencies[i]<max)
            {
                ////System.out.println(frequencies[i]+" included");
                arrayLength++;
            }
        }
        double[][] pmf=new double[stft.length][arrayLength];
        for (int i = 0; i < stft.length; i++)
        {
            double[] spectrum=new double[arrayLength];
            double sumSpectrum=0;
            int spectrumNum=0;
            for(int j=0;j<stft[i].length/2;j++)
            {
                if(frequencies[j]>min&&frequencies[j]<max) {
                    ////System.out.println(frequencies[j]+" reading");
                    spectrum[spectrumNum] = complexMod(stft[i][j]);
                    ////System.out.println(spectrum[spectrumNum]);
                    sumSpectrum += spectrum[spectrumNum];
                    ////System.out.println(sumSpectrum);
                    spectrumNum++;
                    ////System.out.println(spectrumNum);
                }
            }
            spectrumNum=0;
            for(int j=0;j<stft[i].length/2;j++)
            {
                if((frequencies[j]>min&&frequencies[j]<max)) {
                    if (sumSpectrum > 0)
                        pmf[i][spectrumNum] = spectrum[spectrumNum] / sumSpectrum;
                    else
                        pmf[i][spectrumNum]=0;
                    spectrumNum++;
                }
            }
            pmf[i]=normaliseSpectrum(pmf[i]);
        }
        return pmf;
    }
    public static double[] pmf(Complex[] istft)
    {
        double[] pmf=new double[istft.length];
        double sumAmplitude=0;
        for(int i=0;i<istft.length;i++)
        {
            sumAmplitude+=complexMod(istft[i]);
        }
        for(int i=0;i<istft.length;i++)
        {
            pmf[i]=complexMod(istft[i])/sumAmplitude;
            ////System.out.println("PMF "+i+": "+pmf[i]);
        }
        pmf=normaliseSpectrum(pmf);
        return pmf;
    }
    public static double[] normaliseSpectrum(double[] spectrum)
    {
        double sum=0;
        for(int i=0;i<spectrum.length;i++)
        {
            sum+=spectrum[i];
        }
        for(int i=0;i<spectrum.length;i++)
        {
            if(sum>0)
                spectrum[i]/=sum;
            ////System.out.println("Sprectrum "+i+" "+spectrum[i]);
        }
        return spectrum;
    }
    public static double[] backgroundNoise(File wavFile)
    {
        double[] values=getValues(getEnvelope(wavFile));
        double[] bounds=getBounds(findMinValue(values));
        double stdDevFactor=1;
        int[] histogram=new int[100];
        for(int i=0;i<values.length;i++)
        {
            addToHistogram(values[i],bounds,histogram);
        }
        double mode=0;
        int modeCount=0;
        histogram=smoothHistogram(histogram,5);
        for(int j=0;j<100;j++)
        {
            if(histogram[j]>modeCount) {
                modeCount = histogram[j];
                mode=bounds[j];
            }
            ////System.out.println(bounds[j]+": "+histogram[j]);
        }
        ////System.out.println("Mode is "+mode);
        double standardDeviation=standardDeviation(histogram,bounds,mean(histogram,bounds));
        double[] returnVal={mode+stdDevFactor*standardDeviation,standardDeviation};
        new File(wavFile.getPath()+"_envelope.wav").delete();
        return returnVal;
        /*for(int i=0;i<values.length;i++)
        {
            //System.out.println(values[i]);
        }*/
    }
    public static double aci(Complex[][] stft)
    {
        double[] aciSpectrum=aciSpectrum(stft);
        double sumAciSpectrum=0;
        for(int i=0;i<aciSpectrum.length;i++)
        {
            sumAciSpectrum+=aciSpectrum[i];
        }
        return sumAciSpectrum/aciSpectrum.length;
    }

    public static double[] aciSpectrum(Complex[][] stft)
    {
        double[] sumDifferences=new double[stft[0].length/2];
        double[] sumIntensities=new double[stft[0].length/2];
        double[] aciSpectrum=new double[stft[0].length/2];
        double sumAciSpectrum=0;
        for(int i=0;i<stft[0].length/2;i++) {
            sumDifferences[i] = 0;
            sumIntensities[i] = 0;
            for (int j = 0; j < (stft.length - 1); j++) {
                sumDifferences[i] += Math.abs(complexMod(stft[j][i]) - complexMod(stft[(j + 1)][i]));
                /*if(Math.abs(complexMod(stft[j][i])-complexMod(stft[(j+1)][i]))>0.03)
                    //System.out.println((i*128)+" "+(double)j/(22050/128)+": "+Math.abs(complexMod(stft[j][i])-complexMod(stft[(j+1)][i])));
                */
                sumIntensities[i] += complexMod(stft[j][i]);
            }
            sumIntensities[i] += complexMod(stft[stft.length - 1][i]);
            if (sumIntensities[i] > 0)
                aciSpectrum[i] = sumDifferences[i] / sumIntensities[i];
            else
                aciSpectrum[i] = 0;
        }
        return aciSpectrum;
    }

    public static double spectralCover(Complex[][] stft,double[] frequencies,int min,int max,double threshold)
    {
        int totalLooked=0;
        int totalOver=0;
        int wSize=stft[0].length/2;
        for(int i=0;i<stft.length;i++) {
            for (int j = 0; j < wSize; j++) {
                if(frequencies[j]>min&&frequencies[j]<max)
                {
                    totalLooked++;
                    if(complexMod(stft[i][j])>threshold)
                        totalOver++;
                }
            }
        }
        return (double)totalOver/totalLooked;
    }

    public static double [][] mfccs(Complex[][]fft,double[] frequencies,int number,int min,int max)
    {
        double[] melFrequencies=new double[number];
        double minMel=hzToMels(/*1000*/min);
        double maxMel=hzToMels(/*11025*/max);
        double[] energies=new double[number];
        double[] firstMFCCs;
        double[][]allMFCCs=new double[3][number];       //1 - 1st order delta, 2 - second order delta
        double melGap=(maxMel-minMel)/number;
        for(int i=0;i<number;i++)
        {
            melFrequencies[i]=minMel+melGap*i;
        }
        for(int k=0;k<number;k++) {
            energies[k]=0;
            for (int i = 0; i < fft.length; i++) {
                for (int j = 0; j < fft[0].length / 2; j++) {
                    double factor=Math.max(1-Math.abs(hzToMels(frequencies[j])-melFrequencies[k])/(melGap*2),0);
                    energies[k]+=complexMod(fft[i][j])*factor;
                }
            }
            energies[k]=Math.log(energies[k]);
        }
        firstMFCCs= new FastCosineTransformer(DctNormalization.STANDARD_DCT_I).transform(energies, TransformType.FORWARD);
        for(int i=0;i<number;i++)
        {
            allMFCCs[0][i]=firstMFCCs[i];
            if(i!=number-1&&i!=0)
                allMFCCs[1][i-1]=firstMFCCs[i+1]-firstMFCCs[i-1];
        }
        for(int i=1;i<number-2;i++)
        {
            allMFCCs[2][i-1]=allMFCCs[1][i+1]-allMFCCs[1][i-1];
        }
        return allMFCCs;
    }
    public static double hzToMels(double frequency)
    {
        return 1127*Math.log(1+frequency/700);
    }
    public static double melsToHz(double frequency)
    {
        return 770*(Math.exp(frequency/1127)-1);
    }
    public static double convertToDB(short value,int bitRate)
    {
        if(value==0)
            return 1;
        else
            return 20*Math.log10(value/Math.pow(2,(bitRate-1)));
    }
    public static void addToHistogram(double dbValue,double[] bounds,int[] histogram)
    {
        int lastValue=-1;
        boolean set=false;
        ////System.out.println(dbValue);
        for(int i=0;i<100&&!set;i++)
        {
            if(bounds[i]>dbValue)
            {
                histogram[lastValue]++;
                set=true;
            }
            lastValue=i;
        }
    }
    public static double findMinValue(double[] values)
    {
        double min=0;
        for(int i=0;i<values.length;i++)
        {
            if(values[i]<min)
                min=values[i];
        }
        return min;
    }
    public static double[] getValues(File toStream)
    {

        try {
            FileInputStream fileStream = new FileInputStream(toStream);
            InputStream buffer = new BufferedInputStream(fileStream);
            AudioInputStream stream = AudioSystem.getAudioInputStream(buffer);
            int fileSize=(int)toStream.length();
            AudioFormat format = stream.getFormat();         //Format data
            int channels = format.getChannels();          //Number of channels
            int frameSize = format.getFrameSize();        //Size per frame
            int byteSize=frameSize/channels;
            int bitRate=byteSize*8;
            String filePath=toStream.getPath();
            double[] frames = new double[fileSize];
            byte[][] channelBytes = new byte[channels][byteSize];
            for (int i = 0; i < frames.length / channels; i++) {
                byte[] tempBytes = new byte[byteSize * channels];
                stream.read(tempBytes);
                for (int j = 0; j < byteSize * channels; j++) {
                    channelBytes[j / byteSize][j % byteSize] = tempBytes[j];
                }
                for (int k = 0; k < channels; k++) {
                    frames[i * channels + k] = convertToDB(fromByteArray(channelBytes[k]),bitRate);
                }
            }
            stream.close();
            buffer.close();
            fileStream.close();
            return frames;
        }
        catch(IOException e){e.printStackTrace(); return null;}
        catch(UnsupportedAudioFileException f){f.printStackTrace(); return null;}
    }
    public static double[] getNoDBValues(File toStream)
    {

        try {
            FileInputStream fileStream = new FileInputStream(toStream);
            InputStream buffer = new BufferedInputStream(fileStream);
            AudioInputStream stream = AudioSystem.getAudioInputStream(buffer);
            int fileSize=(int)toStream.length();
            AudioFormat format = stream.getFormat();         //Format data
            int channels = format.getChannels();          //Number of channels
            int frameSize = format.getFrameSize();        //Size per frame
            int byteSize=frameSize/channels;
            int bitRate=byteSize*8;
            String filePath=toStream.getPath();
            double[] frames = new double[fileSize/byteSize];
            byte[][] channelBytes = new byte[channels][byteSize];
            for (int i = 0; i < frames.length / channels; i++) {
                byte[] tempBytes = new byte[byteSize * channels];
                stream.read(tempBytes);
                for (int j = 0; j < byteSize * channels; j++) {
                    channelBytes[j / byteSize][j % byteSize] = tempBytes[j];
                }
                for (int k = 0; k < channels; k++) {
                    frames[i * channels + k] = fromByteArray(channelBytes[k])/Math.pow(2,(bitRate-1));
                }
            }
            stream.close();
            buffer.close();
            fileStream.close();
            return frames;
        }
        catch(IOException e){e.printStackTrace(); return null;}
        catch(UnsupportedAudioFileException f){f.printStackTrace(); return null;}
    }
    public static short fromByteArray(byte[] bytes) {
        return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).getShort();
    }
    public static byte[] toByteArray(short value,int byteSize) {
        return ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort(value).array();
    }
    public static File getEnvelope(File toStream)
    {
        try {
            FileInputStream fileStream = new FileInputStream(toStream);
            InputStream buffer = new BufferedInputStream(fileStream);
            AudioInputStream stream = AudioSystem.getAudioInputStream(buffer);
            int fileSize=(int)toStream.length();
            AudioFormat format = stream.getFormat();         //Format data
            int channels = format.getChannels();          //Number of channels
            int frameSize = format.getFrameSize();        //Size per frame
            int byteSize=frameSize/channels;
            String filePath=toStream.getPath();
            byte[] newBytes = new byte[fileSize];
            byte[][] channelBytes = new byte[channels][byteSize];
            for (int i = 0; i < fileSize/byteSize/channels; i++) {
                byte[] tempBytes = new byte[byteSize * channels];
                stream.read(tempBytes);
                for (int j = 0; j < byteSize * channels; j++) {
                    channelBytes[j / byteSize][j % byteSize] = tempBytes[j];
                }
                for (int k = 0; k < channels; k++) {
                    short frame = (short) Math.abs(fromByteArray(channelBytes[k]));
                    ////System.out.println("New Frame: "+frame);
                    byte[] toAdd = toByteArray(frame, 2);
                    ////System.out.println((i * 2 * (k+1))+","+(i * 2 * (k+1) + 1));
                    newBytes[i * 2 * (k+1) + k * 2] = toAdd[0];
                    newBytes[i * 2 * (k+1) + 1 + k * 2] = toAdd[1];
                }
            }
            ////System.out.println("Opening fos");
            FileOutputStream fos = new FileOutputStream(filePath + "_temp.wav");
            fileStream.close();
            fileStream = new FileInputStream(toStream);
            for (int i = 0; i < 44; i++) {
                fos.write(fileStream.read());
            }
            ////System.out.println("Header written");
            fos.write(newBytes);
            ////System.out.println("File written");
            fos.close();
            stream.close();
            buffer.close();
            fileStream.close();
            ExternalProcessExecutor.execute("sox " + filePath + "_temp.wav" + " " + filePath + "_envelope.wav sinc -50");
            new File(filePath + "_temp.wav").delete();
            return new File(filePath + "_envelope.wav");
        }
        catch(IOException e){e.printStackTrace(); return null;}
        catch(UnsupportedAudioFileException f){f.printStackTrace(); return null;}
    }
    public static double[] getBounds(double min)
    {
        double[] bounds=new double[100];
        for(int i=0;i<100;i++)
        {
            bounds[i]=min+0.2*i;
            ////System.out.println("Bounds "+i+": "+bounds[i]);
        }
        return bounds;
    }
    public static int[] smoothHistogram(int[] histogram,int windowSize)
    {
        int[] newHistogram=new int[histogram.length];
        ArrayDeque<Integer> windowQueue=new ArrayDeque<>();
        int sum=0;
        for(int i=0;i<windowSize;i++)
        {
            newHistogram[i]=0;
        }
        for(int i=0;i<histogram.length;i++)
        {
            windowQueue.add(histogram[i]);
            sum+=histogram[i];
            if(i>=windowSize)
            {
                newHistogram[i]=sum/windowSize;
                sum-=windowQueue.poll();
            }
        }
        return newHistogram;
    }
    public static double[] reduceVolume(double[] values,double mode)
    {
        for(int i=0;i<values.length;i++)
        {
            if(values[i]<0) {
                values[i] -= mode;
                if (values[i] < 0)
                    values[i] = 0;
            }
            else
                values[i]=0;
        }
        return values;
    }
    public static double getMinPSD(double[] psds)
    {
        double min=200;
        for(int i=0;i<5;i++)
        {
            double normPSD=psds[i]/RAIN_PSD[i];
            if(normPSD<min)
                min=normPSD;
        }
        return min*10000;
    }

    public static void writeIndiciesSet(File[] filesToWrite,BufferedWriter writer,String cicada,String rain,boolean hasHPF,boolean isDownsampled)
    {
        for(int i=0;i<filesToWrite.length;i++) {
            String name=filesToWrite[i].getPath();
            String newName=name.replaceAll(" ","");
            filesToWrite[i].renameTo(new File(newName));
            filesToWrite[i]=new File(newName);
            long time1;
            long time2;
            int subsetLength=5;
            int sampleRate=22050;
            int min=1000;
            int max=10000;
            int adder=0;
            if(!hasHPF) {
                subsetLength +=2;
                min=0;
                adder=2;
            }
            if(!isDownsampled)
            {
                sampleRate=44100;
                max=20000;
                subsetLength+=1;
            }
            double[] psdSubset=new double[subsetLength];
            double[] bedoyaPsdSubset=new double[subsetLength];
            double[] specEntropySubset=new double[subsetLength];
            double[] aciSubset=new double[subsetLength];
            double[] snrSubset=new double[subsetLength];
            double[] segSNRsubset=new double[subsetLength];
            double[] lowCoverSubset=new double[subsetLength];
            double[] highCoverSubset=new double[subsetLength];
            double[][] mfccs=null;
            time1=System.currentTimeMillis();
            Complex[][] fft=AcousticIndices.stft(filesToWrite[i],256);
            time2=System.currentTimeMillis();
            System.out.println("STFT Time: "+(time2-time1));
            double[] frequencies=AcousticIndices.getFreqeuencies(256,sampleRate);
            time1=System.currentTimeMillis();
            //System.out.println("GetFrequencies Time: "+(time1-time2));
            double temporalEntropy=AcousticIndices.temporalEntropy(fft);
            time2=System.currentTimeMillis();
            //System.out.println("Temporal Entropy Time: "+(time2-time1));
            double[] backgroundNoise=AcousticIndices.backgroundNoise(filesToWrite[i]);
            //time1=System.currentTimeMillis();
            //System.out.println("Background Noise Time: "+(time1-time2));
            double[] psd=AcousticIndices.powerSpectralDensity(fft,fft.length,256);
            time2=System.currentTimeMillis();
            //System.out.println("PSD Time: "+(time2-time1));
            double allPSD=AcousticIndices.frequencySubset(psd, frequencies, min,max);
            //time1=System.currentTimeMillis();
            //System.out.println("All PSD Time: "+(time1-time2));
            double segSNR=AcousticIndices.segmentalSNR(fft,frequencies,min,max,0.1,sampleRate);
            //System.out.println("fft length: "+fft.length);
            double allSNR=AcousticIndices.signalToNoise(fft, fft.length, frequencies, allPSD, 256, min,max);
            time2=System.currentTimeMillis();
            double allBedoyaSNR=AcousticIndices.freqSNR(frequencies,psd,allPSD,min,max);
            //System.out.println("All SNR Time: "+(time2-time1));
            double allSpectralEntropy=AcousticIndices.spectralEntropy(fft);
            //time1=System.currentTimeMillis();
            //System.out.println("All Spectral Entropy Time: "+(time1-time2));
            double allACI=AcousticIndices.aci(fft);
            double[] aciSpectrum=AcousticIndices.aciSpectrum(fft);
            double lowSpectralCover=AcousticIndices.spectralCover(fft,frequencies,min,max,0.0001);
            double highSpectralCover=AcousticIndices.spectralCover(fft,frequencies,min,max,0.0003);
            time2=System.currentTimeMillis();
            System.out.println("Indices Time: "+(time2-time1));
            if(!hasHPF)
            {
                long hpfExtraTime=System.currentTimeMillis();
                for(int j=0;j<2;j++)
                {
                    psdSubset[j] = AcousticIndices.frequencySubset(psd, frequencies, j*500, (j+1)*500);
                    time1=System.currentTimeMillis();
                    //System.out.println("PSD Subset Time: "+(time1-time2));
                    snrSubset[j] = AcousticIndices.signalToNoise(fft, fft.length, frequencies, psdSubset[j], 256, j*500, (j+1)*500);
                    time2=System.currentTimeMillis();
                    //System.out.println("SNR Subset Time: "+(time2-time1));
                    bedoyaPsdSubset[j]=AcousticIndices.freqSNR(frequencies,psd,psdSubset[j],j*500,(j+1)*500);
                    segSNRsubset[j]=AcousticIndices.segmentalSNR(fft,frequencies,j*500,(j+1)*500,0.1,sampleRate);
                    specEntropySubset[j] = AcousticIndices.spectralEntropySubset(fft,frequencies,j*500, (j+1)*500);
                    time1=System.currentTimeMillis();
                    //System.out.println("SpecEntropySubset Time: "+(time1-time2));
                    aciSubset[j] = AcousticIndices.frequencySubset(aciSpectrum,frequencies,j*500, (j+1)*500);
                    time2=System.currentTimeMillis();
                    //System.out.println("ACI Subset Time: "+(time2-time1));
                    lowCoverSubset[j]=AcousticIndices.spectralCover(fft,frequencies,j*500, (j+1)*500,0.0001);
                    highCoverSubset[j]=AcousticIndices.spectralCover(fft,frequencies,j*500, (j+1)*500,0.0003);
                }
                System.out.println("HPF Extra: "+(System.currentTimeMillis()-hpfExtraTime));
            }
            long otherIndicesTime=System.currentTimeMillis();
            for(int j=0+adder;j<5+adder;j++) {
                psdSubset[j] = AcousticIndices.frequencySubset(psd, frequencies, (j-adder)*2000+1000, (j-adder)*2000+3000);
                time1=System.currentTimeMillis();
                //System.out.println("PSD Subset Time: "+(time1-time2));
                snrSubset[j] = AcousticIndices.signalToNoise(fft, fft.length, frequencies, psdSubset[j], 256, (j-adder)*2000+1000, (j-adder)*2000+3000);
                time2=System.currentTimeMillis();
                //System.out.println("SNR Subset Time: "+(time2-time1));
                bedoyaPsdSubset[j]=AcousticIndices.freqSNR(frequencies,psd,psdSubset[j],(j-adder)*2000+1000,(j-adder)*2000+3000);
                segSNRsubset[j]=AcousticIndices.segmentalSNR(fft,frequencies,(j-adder)*2000+1000,(j-adder)*2000+3000,0.1,sampleRate);
                specEntropySubset[j] = AcousticIndices.spectralEntropySubset(fft,frequencies,(j-adder)*2000+1000,(j-adder)*2000+3000);
                time1=System.currentTimeMillis();
                //System.out.println("SpecEntropySubset Time: "+(time1-time2));
                aciSubset[j] = AcousticIndices.frequencySubset(aciSpectrum,frequencies,(j-adder)*2000+1000,(j-adder)*2000+3000);
                time2=System.currentTimeMillis();
                //System.out.println("ACI Subset Time: "+(time2-time1));
                lowCoverSubset[j]=AcousticIndices.spectralCover(fft,frequencies,(j-adder)*2000+1000,(j-adder)*2000+3000,0.0001);
                highCoverSubset[j]=AcousticIndices.spectralCover(fft,frequencies,(j-adder)*2000+1000,(j-adder)*2000+3000,0.0003);
            }
            System.out.println("other indices: "+(System.currentTimeMillis()-otherIndicesTime));
            if(!isDownsampled)
            {
                psdSubset[subsetLength-1] = AcousticIndices.frequencySubset(psd, frequencies, 11000,20000);
                time1=System.currentTimeMillis();
                //System.out.println("PSD Subset Time: "+(time1-time2));
                snrSubset[subsetLength-1] = AcousticIndices.signalToNoise(fft, fft.length, frequencies, psdSubset[subsetLength-1], 256, 11000,20000);
                time2=System.currentTimeMillis();
                bedoyaPsdSubset[subsetLength-1]=AcousticIndices.freqSNR(frequencies,psd,psdSubset[subsetLength-1],11000,20000);
                //System.out.println("SNR Subset Time: "+(time2-time1));
                segSNRsubset[subsetLength-1]=AcousticIndices.segmentalSNR(fft,frequencies,11000,20000,0.1,sampleRate);
                specEntropySubset[subsetLength-1] = AcousticIndices.spectralEntropySubset(fft,frequencies,11000,20000);
                time1=System.currentTimeMillis();
                //System.out.println("SpecEntropySubset Time: "+(time1-time2));
                aciSubset[subsetLength-1] = AcousticIndices.frequencySubset(aciSpectrum,frequencies,11000,20000);
                time2=System.currentTimeMillis();
                //System.out.println("ACI Subset Time: "+(time2-time1));
                lowCoverSubset[subsetLength-1]=AcousticIndices.spectralCover(fft,frequencies,11000,20000,0.0001);
                highCoverSubset[subsetLength-1]=AcousticIndices.spectralCover(fft,frequencies,11000,20000,0.0003);
            }
            long mfccsTime=System.currentTimeMillis();
            mfccs=AcousticIndices.mfccs(fft,frequencies,33,min,max);
            System.out.println("MFCCs time: "+(System.currentTimeMillis()-mfccsTime));
            try {
                writer.write(filesToWrite[i].getName()+ ","+allPSD*1000000 + "," + allSNR + "," + allBedoyaSNR+","+segSNR+","+allSpectralEntropy + "," + temporalEntropy + "," + backgroundNoise[0] + "," + backgroundNoise[1] + "," + allACI + ","+lowSpectralCover+","+highSpectralCover+",");
                for(int k=0;k<subsetLength;k++)
                {
                    writer.write(psdSubset[k]*1000000+","+snrSubset[k]+","+bedoyaPsdSubset[k]+","+segSNRsubset[k]+","+specEntropySubset[k]+","+aciSubset[k]+","+lowCoverSubset[k]+","+highCoverSubset[k]+",");
                }
                for(int j=0;j<33;j++)
                {
                    writer.write(mfccs[0][j]+",");
                }
                for(int j=0;j<31;j++)
                {
                    writer.write(mfccs[1][j]+",");
                }
                for(int j=0;j<29;j++)
                {
                    writer.write(mfccs[2][j]+",");
                }
                writer.write(getMinPSD(psdSubset)+",");
                writer.write(cicada+","+rain);
                writer.newLine();
            }
            catch(IOException e){e.printStackTrace();}
        }
    }
	public static double standardDeviation(int[] histogram,int[] bounds,double mean)
    {
        double sumDiff=0;
        int numValues=0;
        for(int i=0;i<histogram.length;i++)
        {
            sumDiff+=histogram[i]*Math.pow((bounds[i]-mean),2);
            numValues+=histogram[i];
        }
        return Math.pow(sumDiff/(numValues+1),0.5);
    }
    public static double standardDeviation(int[] histogram,double[] bounds,double mean)
    {
        double sumDiff=0;
        int numValues=0;
        for(int i=0;i<histogram.length;i++)
        {
            sumDiff+=histogram[i]*Math.pow((bounds[i]-mean),2);
            numValues+=histogram[i];
        }
        return Math.pow(sumDiff/(numValues+1),0.5);
    }
    public static double mean(int[] histogram,int[] bounds)
    {
        int sum=0;
        int numValues=0;
        for(int i=0;i<histogram.length;i++)
        {
            sum+=histogram[i]*bounds[i];
            numValues+=histogram[i];
        }
        //System.out.println("Mean: "+(double)sum/numValues);
        return (double)sum/numValues;
    }
    public static double mean(int[] histogram,double[] bounds)
    {
        int sum=0;
        int numValues=0;
        for(int i=0;i<histogram.length;i++)
        {
            sum+=histogram[i]*bounds[i];
            numValues+=histogram[i];
        }
        //System.out.println("Mean: "+(double)sum/numValues);
        return (double)sum/numValues;
    }
}
