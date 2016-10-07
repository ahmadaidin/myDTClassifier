/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier;

import java.io.File;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Instances;

/**
 *
 * @author toshiba
 */
public class MyClassifier {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        
        WekaAccessor access = new WekaAccessor();
        Instances train_data = access.loadArff("weather.nominal.arff");
        train_data.toString();
        train_data.firstInstance().toString();
        MyId3 tree1 = new MyId3();
        MyC45 tree2 = new MyC45();
        J48 tree3 = access.buildC45Classifier(train_data);
        Id3 tree4 = access.buildId3Classifier(train_data);
        tree1.buildClassifier(train_data);
        tree2.buildClassifier(train_data);
        System.out.println("=== My Id3 ===");
        access.tenFoldCrossValidation(tree1, train_data);
        System.out.println("=== My C45 ===");
        access.tenFoldCrossValidation(tree2, train_data);
        System.out.println("=== Weka C45 ===");
        access.tenFoldCrossValidation(tree3, train_data);
        System.out.println("=== Weka Id3 ===");
        access.tenFoldCrossValidation(tree4, train_data);
    }
    
}
