import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { ModulesRoutingModule } from './modules-routing.module';
import { HomeComponent } from './home/home.component';
import { HttpClientModule } from '@angular/common/http';


@NgModule({
  declarations: [
    HomeComponent
  ],
  imports: [
    CommonModule,
    ModulesRoutingModule,
    HttpClientModule
  ]
})
export class ModulesModule { }
